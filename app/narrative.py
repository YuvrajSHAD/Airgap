"""Local LLM summary (Ollama) with a deterministic template fallback and number guardrail."""
import json, os, re
import numpy as np
import requests

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")


def llm_available():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=1.0)
        return r.status_code == 200 and any(
            m.get("name") in (LLM_MODEL, LLM_MODEL + ":latest") for m in r.json().get("models", []))
    except Exception:
        return False


def _tier(score, cut):
    return 0 if score < cut[0] else (1 if score < cut[1] else 2)


def facts(res):
    uc, rows, m = res["usecase"], res["rows"], res["metrics"]
    F = {"use_case": uc, "known_rows": res["n_labeled"], "new_rows": res["n_new"], "target": res["target"],
         "top_drivers": [x["feature"] for x in res["importance"][:5]]}
    pool = [r for r in rows if r["src"] == "new"] or rows
    if res["regression"]:
        F.update(mean_yield=round(m["target_mean"], 1), mae=round(m["mae"], 2), r2=round(m["r2"], 2),
                 interval_coverage_pct=round(m["coverage"] * 100), below_expected=m["below"], above_expected=m["above"])
        flagged = sorted([r for r in rows if r.get("flag") == "below"], key=lambda r: r["actual"] - r["lo"])[:5]
        F["worst_batches"] = [{"id": r["id"], "actual": r["actual"], "expected_low": r["lo"], "expected": r["score"],
                               "drivers": [d["f"] for d in r["d"]]} for r in flagged]
        return F
    cut = res["cutoffs"]
    tiers = [0, 0, 0]
    for r in pool:
        tiers[_tier(r["score"], cut)] += 1
    F.update(auc=round(m["auc"], 2), base_rate_pct=round(m["base_rate"] * 100),
             lift_top_decile=round(m["lift_top_decile"] or 0, 1), tier_counts=dict(zip(
                 {"clinical": ["low", "moderate", "high"], "industrial": ["normal", "watch", "likely_failure"],
                  "credit": ["approve", "review", "decline"]}[uc], tiers)), scored_group="new rows" if res["n_new"] else "all rows")
    top = sorted(pool, key=lambda r: -r["score"])[:5]
    F["highest_risk"] = [{"id": r["id"], "risk_pct": round(r["score"] * 100),
                          "drivers": [f'{d["f"]}={d["v"]}' for d in r["d"]]} for r in top]
    return F


def template(res, F):
    uc = F["use_case"]
    drv = ", ".join(F["top_drivers"][:4])
    if res["regression"]:
        p1 = (f"{F['known_rows']} batches with known {F['target']} were used as context. Average {F['target']} is {F['mean_yield']}. "
              f"Cross-validated error is {F['mae']} on average (R2 {F['r2']}), and {F['interval_coverage_pct']}% of actual values fell inside the predicted 80% range.")
        p2 = f"{F['below_expected']} batches came in below their predicted range and {F['above_expected']} above it. Variables that move the outcome most: {drv}."
        if F["worst_batches"]:
            w = F["worst_batches"][0]
            p3 = f"Largest shortfall: batch {w['id']} reached {w['actual']} against an expected {w['expected']}. Inspect {', '.join(w['drivers']) or 'its sensor log'} first."
        else:
            p3 = "No batch fell below its predicted range."
        return [p1, p2, p3]
    t = F["tier_counts"]
    k = list(t)
    head = {"clinical": "Clinical summary", "industrial": "Reliability summary", "credit": "Portfolio summary"}[uc]
    p1 = (f"{head}. {F['known_rows']} records with known outcomes were used as context (event rate {F['base_rate_pct']}%). "
          f"Cross-validated AUC is {F['auc']}, and the top decile of scores holds {F['lift_top_decile']} times the base event rate.")
    p2 = (f"Scored {F['scored_group']}: {t[k[2]]} {k[2].replace('_', ' ')}, {t[k[1]]} {k[1]}, {t[k[0]]} {k[0]}. "
          f"Strongest signals in the data: {drv}.")
    hr = F["highest_risk"][0] if F["highest_risk"] else None
    p3 = (f"Highest score: {hr['id']} at {hr['risk_pct']}%, driven by {', '.join(hr['drivers']) or 'no single dominant feature'}." if hr else "")
    tail = {"clinical": "Decision support for research use only. A clinician must confirm any action.",
            "industrial": "Treat scores as maintenance prioritisation, not a safety certification.",
            "credit": "Scores are decision support. Reason codes above should accompany any adverse action notice. Review for fair-lending compliance before use."}[uc]
    return [p for p in (p1, p2, p3, tail) if p]


def _bad_number(text, F):
    """Return the first number in the text that is not in the facts, else None."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)          # 7,800 -> 7800
    allowed = set(re.findall(r"\d+(?:\.\d+)?", json.dumps(F)))
    for n in re.findall(r"\d+(?:\.\d+)?", text):
        if n in allowed or (n.isdigit() and int(n) <= 10) or n.rstrip("0").rstrip(".") in allowed:
            continue
        return n
    return None


def _numbers_ok(text, F):
    return _bad_number(text, F) is None


def summarize(res):
    F = facts(res)
    style = {"clinical": "a one-page clinical summary for a lab clinician",
             "industrial": "a short shift report for a plant engineer",
             "credit": "a short portfolio memo for a loan officer"}[F["use_case"]]
    reason = None
    if not llm_available():
        reason = f"Ollama not reachable or model {LLM_MODEL} not installed"
    else:
        prompt = (f"Write {style}. Use ONLY the facts in the JSON. Do not invent numbers, names, diagnoses or causes. "
                  f"Do not mention accuracy. Only use the tier names given in tier_counts, and never call a tier that has 0 cases present. "
                  f"3 short paragraphs, plain text, no markdown.\nFACTS:\n{json.dumps(F)}")
        try:
            r = requests.post(f"{OLLAMA}/api/chat", timeout=300, json={
                "model": LLM_MODEL, "stream": False, "think": False, "options": {"temperature": 0.2},
                "messages": [{"role": "user", "content": prompt}]})
            data = r.json()
            if "message" not in data:
                reason = f"Ollama error: {data.get('error', data)}"
            else:
                text = data["message"]["content"].strip()
                # safety net: drop any <think>...</think> block if the model emits one anyway
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                bad = _bad_number(text, F) if text else None
                if not text:
                    reason = "model returned an empty reply"
                elif bad is not None:
                    reason = f"guardrail rejected the text: number {bad} is not in the results"
                else:
                    return {"paragraphs": [p.strip() for p in text.split("\n\n") if p.strip()],
                            "source": f"local LLM ({LLM_MODEL}), numbers verified"}
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
    print(f"[narrative] LLM summary not used: {reason}", flush=True)
    return {"paragraphs": template(res, F), "source": f"built-in template (LLM not used: {reason})"}
