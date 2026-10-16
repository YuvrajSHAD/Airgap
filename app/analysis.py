"""Use-case pipeline: CSV/XLSX -> TabPFN scores, uncertainty, drivers."""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .engine import Engine, MAX_TRAIN_ROWS

POS_WORDS = {"1", "true", "yes", "y", "high", "default", "defaulted", "positive", "fail", "failure",
             "failed", "readmitted", "toxic", "bad", "event"}
USECASES = {
    "clinical": {"tiers": ["Low", "Moderate", "High"], "label": "Clinical risk"},
    "industrial": {"tiers": ["Normal", "Watch", "Likely failure"], "label": "Industrial yield and failure"},
    "credit": {"tiers": ["Approve", "Review", "Decline"], "label": "Credit decision"},
}


class AnalysisError(Exception):
    pass


def _norm(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    return str(v).strip().lower()


def read_table(name, raw):
    import io
    n = name.lower()
    if n.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
    else:
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
                break
            except Exception:
                df = None
        if df is None:
            raise AnalysisError("Could not parse the file as CSV or Excel.")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) < 20 or df.shape[1] < 3:
        raise AnalysisError("Need at least 20 rows and 3 columns.")
    return df.reset_index(drop=True)


def guess_id(df):
    for c in df.columns:
        lc = c.lower()
        if lc.endswith("_id") or lc in ("id", "patient", "batch", "application", "record"):
            if df[c].nunique() >= 0.9 * len(df):
                return c
    return None


def profile(df, usecase=None):
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({"name": c, "kind": "number" if pd.api.types.is_numeric_dtype(s) else "text",
                     "missing_pct": round(float(s.isna().mean()) * 100, 1), "unique": int(s.nunique())})
    hints = ["readmit", "outcome", "default", "fail", "yield", "target", "label", "toxic", "risk", "status"]
    guess = None
    for h in hints:
        for c in df.columns:
            if h in c.lower() and df[c].isna().mean() < 0.5:
                guess = c; break
        if guess: break
    if guess is None:
        guess = df.columns[-1]
    return {"rows": int(len(df)), "columns": cols, "id_guess": guess_id(df), "target_guess": guess}


def prepare_features(df, drop):
    X = pd.DataFrame(index=df.index)
    cat_idx, notes, cat_levels = [], [], {}
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            notes.append(f"{c}: date column ignored"); continue
        if s.dtype == bool:
            s = s.astype(float)
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce").astype(float)
            is_cat = False
        else:
            num = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() and num.notna().sum() >= 0.9 * s.notna().sum():
                x, is_cat = num.astype(float), False
            else:
                if s.nunique() > max(50, 0.5 * len(s)):
                    notes.append(f"{c}: free text or ID-like, ignored"); continue
                codes, uniq = pd.factorize(s.astype("object"))
                x = pd.Series(codes, index=s.index).astype(float)
                x[codes < 0] = np.nan
                is_cat = True
                cat_levels[c] = list(uniq)
        if x.nunique(dropna=True) < 2:
            notes.append(f"{c}: constant, ignored"); continue
        X[c] = x
        if is_cat:
            cat_idx.append(X.shape[1] - 1)
    if X.shape[1] == 0:
        raise AnalysisError("No usable feature columns.")
    return X, cat_idx, notes


def pick_positive(raw, forced):
    uniq = list(pd.unique(raw.dropna()))
    norm = {_norm(u): u for u in uniq}
    if forced is not None and str(forced).strip().lower() in norm:
        return norm[str(forced).strip().lower()]
    for k in norm:
        if k in POS_WORDS:
            return norm[k]
    counts = raw.map(_norm).value_counts()
    return norm[counts.idxmin()]


def _cap(idx, rng):
    if len(idx) > MAX_TRAIN_ROWS:
        return np.sort(rng.choice(idx, MAX_TRAIN_ROWS, replace=False))
    return idx


def _disp(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if abs(v) >= 1000: return f"{v:,.0f}"
        return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v) >= 10 else f"{v:.3g}"
    return str(v)


def run(df, usecase, target, params, progress):
    t0 = time.time()
    if usecase not in USECASES:
        raise AnalysisError("Unknown use case.")
    if target not in df.columns:
        raise AnalysisError(f"Target column '{target}' not found.")
    eng = Engine()
    rng = np.random.default_rng(0)
    id_col = params.get("id_col") or guess_id(df)
    if id_col == target:
        id_col = None
    ids = df[id_col].astype(str).tolist() if id_col else [str(i + 1) for i in range(len(df))]
    raw_t = df[target].replace(r"^\s*$", np.nan, regex=True)
    lab = raw_t.notna().to_numpy()
    n_lab, n_new = int(lab.sum()), int((~lab).sum())
    if n_lab < 30:
        raise AnalysisError(f"Only {n_lab} rows have a value in '{target}'. Need at least 30 known outcomes.")

    regression = False
    tnum = pd.to_numeric(raw_t, errors="coerce")
    if usecase == "industrial" and tnum[lab].notna().all() and tnum[lab].nunique() > 10:
        regression = True
    X, cat_idx, notes = prepare_features(df, drop={target, id_col} if id_col else {target})
    names = list(X.columns)
    Xv = X.to_numpy(dtype=float)
    li, ui = np.where(lab)[0], np.where(~lab)[0]
    progress(0.04, f"Prepared {len(names)} features, {n_lab} known rows, {n_new} new rows")

    out = {"usecase": usecase, "regression": regression, "target": target, "id_col": id_col,
           "features": names, "notes": notes, "n_labeled": n_lab, "n_new": n_new, "n_rows": len(df)}

    if regression:
        y = tnum.to_numpy(dtype=float)
        yl = y[li]
        k = 5 if n_lab <= 8000 else 3
        mid_oof, lo_oof, hi_oof = np.zeros(n_lab), np.zeros(n_lab), np.zeros(n_lab)
        for f, (tr, te) in enumerate(KFold(k, shuffle=True, random_state=0).split(li)):
            trc = _cap(tr, rng)
            m = eng.regressor(cat_idx, len(trc)).fit(Xv[li[trc]], yl[trc])
            mid_oof[te], lo_oof[te], hi_oof[te] = m.predict(Xv[li[te]])
            progress(0.05 + 0.55 * (f + 1) / k, f"Cross-validating fold {f + 1} of {k}")
        trc = _cap(np.arange(n_lab), rng)
        full = eng.regressor(cat_idx, len(trc)).fit(Xv[li[trc]], yl[trc])
        pm, pl, ph = (np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df)))
        pm[li], pl[li], ph[li] = mid_oof, lo_oof, hi_oof
        if n_new:
            pm[ui], pl[ui], ph[ui] = full.predict(Xv[ui])
        progress(0.65, "Scored new rows")
        resid = yl - mid_oof
        out["metrics"] = {"mae": float(mean_absolute_error(yl, mid_oof)), "r2": float(r2_score(yl, mid_oof)),
                          "coverage": float(np.mean((yl >= lo_oof) & (yl <= hi_oof))),
                          "target_mean": float(yl.mean()), "target_std": float(yl.std()),
                          "below": int((yl < lo_oof).sum()), "above": int((yl > hi_oof).sum())}
        scale = float(yl.std()) or 1.0
        score_all = pm
        order_labeled = li[np.argsort(-np.abs(resid))]
        predict_full = lambda M: full.predict(M)[0]
        eff_thresh = 0.02 * scale
    else:
        pos = pick_positive(raw_t[lab], params.get("positive"))
        yb = np.zeros(len(df), dtype=int)
        yb[li] = (raw_t[lab].map(_norm) == _norm(pos)).astype(int).to_numpy()
        yl = yb[li]
        minority = int(min(yl.sum(), n_lab - yl.sum()))
        if minority < 5:
            raise AnalysisError(f"Need at least 5 rows of each outcome in '{target}'. Found {minority} of the rarer one.")
        k = min(5 if n_lab <= 8000 else 3, minority)
        oof = np.zeros(n_lab)
        for f, (tr, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=0).split(li, yl)):
            trc = _cap(tr, rng)
            m = eng.classifier(cat_idx, len(trc)).fit(Xv[li[trc]], yl[trc])
            oof[te] = m.proba_pos(Xv[li[te]])
            progress(0.05 + 0.55 * (f + 1) / k, f"Cross-validating fold {f + 1} of {k}")
        trc = _cap(np.arange(n_lab), rng)
        full = eng.classifier(cat_idx, len(trc)).fit(Xv[li[trc]], yl[trc])
        score_all = np.zeros(len(df))
        score_all[li] = oof
        if n_new:
            score_all[ui] = full.proba_pos(Xv[ui])
        progress(0.65, "Scored new rows")
        top = np.argsort(-oof)[: max(1, n_lab // 10)]
        base = float(yl.mean())
        out["metrics"] = {"auc": float(roc_auc_score(yl, oof)), "base_rate": base,
                          "accuracy": float(np.mean((oof >= 0.5) == yl)),
                          "brier": float(np.mean((oof - yl) ** 2)),
                          "lift_top_decile": float(yl[top].mean() / base) if base else None,
                          "positive_label": _norm(pos)}
        order_labeled = li[np.argsort(-oof)]
        predict_full = lambda M: full.proba_pos(M)
        eff_thresh = 0.01

    # ---- drivers by occlusion ------------------------------------------
    progress(0.70, "Explaining drivers")
    Xl_f = pd.DataFrame(Xv[li]).fillna(pd.DataFrame(Xv[li]).median())
    yy = (yl if regression else yl)
    try:
        mi = (mutual_info_regression if regression else mutual_info_classif)(
            Xl_f.to_numpy(), yy, random_state=0)
    except Exception:
        mi = np.ones(len(names))
    kf = min(10, len(names))
    top_f = np.argsort(-mi)[:kf]
    fill = np.nanmedian(Xv, axis=0)
    for j in cat_idx:
        col = Xv[:, j][~np.isnan(Xv[:, j])]
        if len(col):
            fill[j] = pd.Series(col).mode().iloc[0]
    fill = np.where(np.isnan(fill), 0.0, fill)
    sel = list(ui[:100]) + list(order_labeled[: max(0, 150 - min(100, n_new))])
    sel = np.array(sel, dtype=int)
    base_pred = predict_full(Xv[sel])
    stack = np.repeat(Xv[sel][None, :, :], len(top_f), axis=0).copy()
    for a, j in enumerate(top_f):
        stack[a, :, j] = fill[j]
    occ = predict_full(stack.reshape(-1, Xv.shape[1])).reshape(len(top_f), len(sel))
    delta = base_pred[None, :] - occ                        # positive = this feature pushes the score up
    imp = np.abs(delta).mean(axis=1)
    mx = imp.max() or 1.0
    out["importance"] = [{"feature": names[j], "score": round(float(imp[a] / mx) * 100, 1)}
                         for a, j in sorted(enumerate(top_f), key=lambda t: -imp[t[0]])]
    drivers = {}
    for r, ridx in enumerate(sel):
        col = np.argsort(-np.abs(delta[:, r]))[:3]
        ds = []
        for a in col:
            if abs(delta[a, r]) < eff_thresh:
                continue
            f = names[top_f[a]]
            ds.append({"f": f, "v": _disp(df.at[ridx, f]), "e": round(float(delta[a, r]), 4)})
        drivers[int(ridx)] = ds
    progress(0.92, "Building report")

    rows = []
    for i in range(len(df)):
        r = {"id": ids[i], "src": "cv" if lab[i] else "new", "score": round(float(score_all[i]), 4),
             "d": drivers.get(i, [])}
        if lab[i]:
            r["actual"] = float(yl[np.searchsorted(li, i)]) if regression else int(yb[i])
        if regression:
            r.update(lo=round(float(pl[i]), 3), hi=round(float(ph[i]), 3))
            if lab[i]:
                a = r["actual"]
                r["flag"] = "below" if a < pl[i] else ("above" if a > ph[i] else None)
            r["score"] = round(float(pm[i]), 3)
        rows.append(r)
    out["rows"] = rows

    sc_l = score_all[li]
    if regression:
        out["cutoffs"] = None
    elif usecase == "credit":
        out["cutoffs"] = [round(float(np.quantile(sc_l, 0.6)), 3), round(float(np.quantile(sc_l, 0.85)), 3)]
    else:
        out["cutoffs"] = [0.2, 0.5]
    out["engine"] = {"name": eng.name, "device": eng.device, "fallback": eng.fallback}
    out["seconds"] = round(time.time() - t0, 2)
    eng.cleanup()
    return out
