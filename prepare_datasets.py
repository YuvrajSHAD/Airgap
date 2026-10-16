"""Download real public datasets (UCI, CC BY 4.0), clean them, and write upload-ready CSVs to data/real/.
Airgap then shows them as "Real data" buttons. Run once while online:  python prepare_datasets.py
Options: --rows 8000 (max rows per file)   --only credit,clinical,ai4i,secom   --force (re-clean from cached downloads)"""
import argparse, json, os, re, sys, urllib.request, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, OUT = os.path.join(HERE, "data", "raw"), os.path.join(HERE, "data", "real")
BASE = "https://archive.ics.uci.edu/static/public/"
SRC = {
    "clinical": ("diabetes130.zip", BASE + "296/diabetes+130-us+hospitals+for+years+1999-2008.zip"),
    "ai4i": ("ai4i.zip", BASE + "601/ai4i+2020+predictive+maintenance+dataset.zip"),
    "secom": ("secom.zip", BASE + "179/secom.zip"),
    "credit": ("taiwan_credit.zip", BASE + "350/default+of+credit+card+clients.zip"),
}


def fetch(key):
    name, url = SRC[key]
    path = os.path.join(RAW, name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    os.makedirs(RAW, exist_ok=True)
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 airgap-prepare"})
    with urllib.request.urlopen(req, timeout=180) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def member(z, *suffixes):
    for n in z.namelist():
        if n.lower().endswith(suffixes):
            return n
    raise FileNotFoundError(f"none of {suffixes} in zip: {z.namelist()}")


def finish(df, outcome, rows, holdout, seed=0):
    """Shuffle, cap rows, blank the outcome on the last `holdout` rows so Airgap also scores 'new' rows."""
    if len(df) > rows:
        df = df.sample(rows, random_state=seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df[outcome] = df[outcome].astype(object)
    df.loc[len(df) - holdout:, outcome] = np.nan
    return df


def clinical(rows, holdout):
    z = zipfile.ZipFile(fetch("clinical"))
    df = pd.read_csv(z.open(member(z, "diabetic_data.csv")), na_values=["?"], low_memory=False)
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]   # died or hospice: cannot be readmitted
    df["readmitted_30d"] = np.where(df["readmitted"] == "<30", "yes", "no")
    df = df.drop(columns=["readmitted", "patient_nbr", "weight", "payer_code"], errors="ignore")
    for c in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        df[c] = df[c].astype(str)
    df = df.rename(columns={"encounter_id": "encounter_id"})
    df["encounter_id"] = "E" + df["encounter_id"].astype(str)
    return finish(df, "readmitted_30d", rows, holdout)


def ai4i(rows, holdout):
    z = zipfile.ZipFile(fetch("ai4i"))
    df = pd.read_csv(z.open(member(z, ".csv")))
    df.columns = [re.sub(r"[^a-z0-9]+", "_", re.sub(r"\s*\[.*?\]", "", c).lower()).strip("_") for c in df.columns]
    leak = ["twf", "hdf", "pwf", "osf", "rnf", "product_id"]      # failure-mode flags ARE the label: dropping them is essential
    df = df.drop(columns=[c for c in leak if c in df.columns])
    df = df.rename(columns={"udi": "record_id"})
    df["record_id"] = "U" + df["record_id"].astype(str)
    return finish(df, "machine_failure", rows, holdout)


def secom(rows, holdout):
    z = zipfile.ZipFile(fetch("secom"))
    X = pd.read_csv(z.open(member(z, "secom.data")), sep=r"\s+", header=None, na_values=["NaN", "nan"])
    L = pd.read_csv(z.open(member(z, "secom_labels.data")), sep=" ", header=None, names=["label", "ts"], quotechar='"')
    miss = X.isna().mean()
    ok = [c for c in X.columns if miss[c] < 0.4 and X[c].nunique() > 5]
    keep = sorted(ok, key=lambda c: miss[c])[:100]                 # label-free choice: best-filled 100 sensors
    df = X[keep].copy()
    df.columns = [f"sensor_{c:03d}" for c in keep]
    df.insert(0, "wafer_id", [f"W{i:04d}" for i in range(len(df))])
    df["yield_test"] = np.where(L["label"].to_numpy() == 1, "fail", "pass")
    return finish(df, "yield_test", rows, holdout)


def credit(rows, holdout):
    z = zipfile.ZipFile(fetch("credit"))
    name = member(z, ".xls", ".xlsx", ".csv")
    raw = (pd.read_csv(z.open(name), header=None) if name.lower().endswith(".csv")
           else pd.read_excel(z.open(name), header=None))             # .xls needs: pip install xlrd
    hdr = next(i for i in range(5) if "LIMIT_BAL" in raw.iloc[i].astype(str).tolist())
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"default payment next month": "defaulted", "pay_0": "pay_1"})
    df["sex"] = df["sex"].map({1: "male", 2: "female"})
    df["education"] = df["education"].map({1: "graduate school", 2: "university", 3: "high school"}).fillna("other")
    df["marriage"] = df["marriage"].map({1: "married", 2: "single"}).fillna("other")
    df["id"] = "A" + df["id"].astype(int).astype(str)
    df = df.rename(columns={"id": "applicant_id"})
    df["defaulted"] = df["defaulted"].astype(int)
    return finish(df, "defaulted", rows, holdout)


JOBS = {
    "clinical": dict(fn=clinical, usecase="clinical", file="diabetes_readmission_REAL.csv", holdout=200,
                     label="Real: hospital readmission", credit="Diabetes 130-US Hospitals, UCI, CC BY 4.0"),
    "ai4i": dict(fn=ai4i, usecase="industrial", file="ai4i_machine_failure.csv", holdout=300,
                 label="Real-style: machine failure (AI4I, synthetic)", credit="AI4I 2020 Predictive Maintenance (synthetic), UCI, CC BY 4.0"),
    "secom": dict(fn=secom, usecase="industrial", file="secom_fab_REAL.csv", holdout=150,
                  label="Real: semiconductor fab (SECOM)", credit="SECOM, UCI, CC BY 4.0"),
    "credit": dict(fn=credit, usecase="credit", file="taiwan_credit_default_REAL.csv", holdout=300,
                   label="Real: credit card defaults (Taiwan)", credit="Default of Credit Card Clients, UCI, CC BY 4.0"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8000)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    mpath = os.path.join(OUT, "manifest.json")
    man = {m["key"]: m for m in json.load(open(mpath))} if os.path.exists(mpath) else {}
    want = [k for k in JOBS if not a.only or k in a.only.split(",")]
    for k in want:
        j = JOBS[k]
        print(f"[{k}] {j['label']}")
        try:
            df = j["fn"](a.rows, j["holdout"])
        except ImportError as e:
            print(f"  FAILED: {e}. Run: pip install xlrd"); continue
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}"); continue
        df.to_csv(os.path.join(OUT, j["file"]), index=False)
        man[k] = dict(key=k, usecase=j["usecase"], file=j["file"], label=j["label"], credit=j["credit"], rows=len(df), columns=df.shape[1])
        print(f"  wrote {j['file']}: {len(df)} rows x {df.shape[1]} columns")
    json.dump(list(man.values()), open(mpath, "w"), indent=1)
    print("\nDone. Restart run.bat and use the 'Real' buttons in Airgap.")


if __name__ == "__main__":
    main()
