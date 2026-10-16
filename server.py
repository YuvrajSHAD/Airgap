"""Airgap: local-only TabPFN dashboard. Binds 127.0.0.1 and keeps uploads in memory."""
import io, os, threading, time, uuid, traceback
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

os.environ.setdefault("TABPFN_NO_BROWSER", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
from app import analysis, engine, narrative, samples

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Airgap")
FILES, JOBS = {}, {}
LOCK = threading.Lock()          # one analysis at a time keeps the GPU predictable


@app.get("/api/health")
def health():
    hw = engine.hardware()
    return {"tabpfn": engine.tabpfn_version(), "fallback_allowed": engine.ALLOW_FALLBACK,
            "llm": narrative.llm_available(), "llm_model": narrative.LLM_MODEL,
            "offline": os.environ.get("HF_HUB_OFFLINE") == "1", **hw}


@app.get("/api/sample/{usecase}")
def sample(usecase: str, variant: str = ""):
    if usecase not in analysis.USECASES:
        raise HTTPException(404)
    df, name = samples.make(usecase, variant)
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


REAL = os.path.join(HERE, "data", "real")


def _manifest():
    p = os.path.join(REAL, "manifest.json")
    if not os.path.exists(p):
        return []
    import json
    return [m for m in json.load(open(p)) if os.path.exists(os.path.join(REAL, m["file"]))]


@app.get("/api/real")
def real_list():
    return _manifest()


@app.get("/api/real/{key}")
def real_file(key: str):
    m = next((m for m in _manifest() if m["key"] == key), None)
    if not m:
        raise HTTPException(404)
    return FileResponse(os.path.join(REAL, m["file"]), media_type="text/csv", filename=m["file"])


@app.post("/api/upload")
async def upload(file: UploadFile):
    raw = await file.read()
    try:
        df = analysis.read_table(file.filename or "data.csv", raw)
    except analysis.AnalysisError as e:
        raise HTTPException(400, str(e))
    fid = uuid.uuid4().hex[:10]
    FILES[fid] = {"df": df, "name": file.filename}
    return {"file_id": fid, "name": file.filename, **analysis.profile(df)}


@app.post("/api/analyze")
def start(b: dict):
    f = FILES.get(b.get("file_id"))
    if not f:
        raise HTTPException(404, "Upload the file again.")
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"state": "queued", "p": 0.0, "msg": "Waiting for the engine"}
    threading.Thread(target=_work, args=(jid, f, b), daemon=True).start()
    return {"job": jid}


def _work(jid, f, b):
    job = JOBS[jid]
    def prog(p, m): job.update(p=round(p, 3), msg=m)
    with LOCK:
        job.update(state="running")
        try:
            res = analysis.run(f["df"], b["usecase"], b["target"], b, prog)
            prog(0.95, "Writing summary")
            res["summary"] = narrative.summarize(res)
            res["file"] = f["name"]
            job.update(state="done", p=1.0, msg="Done", result=res)
        except analysis.AnalysisError as e:
            job.update(state="error", msg=str(e))
        except Exception as e:
            traceback.print_exc()
            job.update(state="error", msg=f"{type(e).__name__}: {e}")


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404)
    return {k: v for k, v in j.items() if k != "result"} | ({"result": j["result"]} if j["state"] == "done" else {})


@app.get("/api/jobs/{jid}/csv")
def csv(jid: str):
    j = JOBS.get(jid)
    if not j or j["state"] != "done":
        raise HTTPException(404)
    r = j["result"]
    rows = []
    for x in r["rows"]:
        row = {"id": x["id"], "source": "new" if x["src"] == "new" else "cross-validated", "score": x["score"]}
        if r["regression"]:
            row.update(expected_low=x["lo"], expected_high=x["hi"], actual=x.get("actual"), flag=x.get("flag"))
        else:
            row["actual"] = x.get("actual")
        row["drivers"] = "; ".join(f'{d["f"]}={d["v"]} ({d["e"]:+.3f})' for d in x["d"])
        rows.append(row)
    return Response(pd.DataFrame(rows).to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="airgap_scored.csv"'})


app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")), log_level="warning")
