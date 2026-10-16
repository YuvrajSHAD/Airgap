"""Synthetic demo data. Every sample is generated here, labelled synthetic, and not real patients, plants or borrowers."""
import numpy as np
import pandas as pd


def _sig(x): return 1 / (1 + np.exp(-x))


def _holes(df, cols, rate, rng):
    for c in cols:
        df.loc[rng.random(len(df)) < rate, c] = np.nan
    return df


def clinical(n=900, new=60, seed=7):
    r = np.random.default_rng(seed); N = n + new
    age = r.normal(62, 14, N).clip(18, 95).round()
    sex = r.choice(["F", "M"], N)
    bmi = r.normal(28, 5.5, N).clip(16, 55).round(1)
    prior = r.poisson(1.2, N)
    creat = (r.lognormal(0.05, 0.35, N) + (age - 60) * 0.004).round(2)
    egfr = (110 - age * 0.7 - creat * 18 + r.normal(0, 6, N)).clip(8, 130).round(0)
    crp = r.lognormal(1.2, 0.9, N).round(1)
    hba1c = (r.normal(6.1, 1.1, N) + (bmi - 28) * 0.03).clip(4.2, 14).round(1)
    sbp = (r.normal(130, 18, N) + (age - 60) * 0.3).round()
    hb = r.normal(13, 1.8, N).round(1)
    wbc = r.lognormal(2.0, 0.28, N).round(1)
    alb = r.normal(4.0, 0.5, N).clip(2, 5.2).round(1)
    ldl = r.normal(115, 32, N).round(); hdl = r.normal(50, 13, N).round()
    trig = r.lognormal(4.9, 0.4, N).round()
    alt = r.lognormal(3.2, 0.5, N).round(); ast = (alt * r.normal(0.9, 0.2, N)).round()
    plt = r.normal(250, 60, N).round()
    smoker = r.choice(["yes", "no"], N, p=[0.22, 0.78])
    gene_a = r.choice([0, 1, 2], N, p=[0.6, 0.3, 0.1]); gene_b = r.choice([0, 1, 2], N, p=[0.7, 0.25, 0.05])
    anticoag = r.choice(["yes", "no"], N, p=[0.2, 0.8])
    z = (-3.1 + 0.03 * (age - 60) + 0.42 * prior + 0.5 * np.log1p(crp) - 0.02 * (egfr - 70)
         + 0.35 * (hba1c > 7.5) + 0.5 * (alb < 3.5) + 0.4 * gene_a + 0.25 * (smoker == "yes")
         + 0.5 * ((gene_a > 0) & (creat > 1.3)) + r.normal(0, 0.5, N))
    y = np.where(r.random(N) < _sig(z), "yes", "no").astype(object)
    df = pd.DataFrame(dict(patient_id=[f"P{1000 + i}" for i in range(N)], age=age, sex=sex, bmi=bmi, systolic_bp=sbp,
                           hba1c=hba1c, creatinine=creat, egfr=egfr, crp=crp, wbc=wbc, hemoglobin=hb, platelets=plt,
                           ldl=ldl, hdl=hdl, triglycerides=trig, albumin=alb, alt=alt, ast=ast,
                           prior_admissions=prior, smoker=smoker, gene_marker_a=gene_a, gene_marker_b=gene_b,
                           on_anticoagulant=anticoag, readmitted_30d=y))
    df = _holes(df, ["crp", "hba1c", "albumin", "alt", "ast", "ldl", "hdl", "triglycerides", "egfr", "gene_marker_b"], 0.08, r)
    df.loc[n:, "readmitted_30d"] = np.nan
    return df


def industrial(kind="yield", n=700, new=50, seed=11):
    r = np.random.default_rng(seed); N = n + new
    machine = r.choice([f"M{i}" for i in range(1, 7)], N)
    shift = r.choice(["A", "B", "C"], N)
    supplier = r.choice(["north", "delta", "kappa"], N, p=[0.5, 0.3, 0.2])
    temp = r.normal(182, 6, N).round(1)
    vib = r.lognormal(0.9, 0.35, N).round(2)
    press = r.normal(6.2, 0.5, N).round(2)
    hum = r.normal(48, 9, N).clip(15, 90).round(0)
    pur = r.normal(97.5, 1.2, N).clip(90, 100).round(2)
    cyc = r.normal(42, 4, N).round(1)
    wear = r.uniform(0, 100, N).round(1)
    amb = r.normal(22, 4, N).round(1)
    mshift = {"M1": 0, "M2": -0.6, "M3": 0.4, "M4": -1.4, "M5": 0.2, "M6": -0.3}
    if kind == "failure":
        z = (-7.4 + 0.9 * vib + 0.045 * wear + 0.08 * (temp - 182) + 0.5 * (press - 6.2) ** 2 * 4
             + np.array([mshift[m] * -1 for m in machine]) * 0.5 + r.normal(0, 0.6, N))
        y = np.where(r.random(N) < _sig(z), "failure", "ok").astype(object); tname = "machine_failure"
    else:
        yld = (96.5 - 0.012 * (temp - 183) ** 2 - 1.1 * np.maximum(vib - 3, 0) - 0.025 * wear
               + 0.55 * (pur - 97.5) - 0.8 * (press - 6.2) ** 2 + np.array([mshift[m] for m in machine])
               - 0.015 * np.maximum(hum - 60, 0) + r.normal(0, 0.55, N))
        bad = r.random(N) < 0.025                         # injected unexplained dips
        yld[bad] -= r.uniform(4, 9, bad.sum())
        y = yld.clip(60, 100).round(2); tname = "yield_pct"
    df = pd.DataFrame(dict(batch_id=[f"B{5000 + i}" for i in range(N)], machine=machine, shift=shift,
                           supplier=supplier, temp_c=temp, vibration_mm_s=vib, pressure_bar=press, humidity_pct=hum,
                           material_purity=pur, cycle_time_s=cyc, tool_wear_pct=wear, ambient_c=amb))
    df[tname] = y
    df = _holes(df, ["temp_c", "pressure_bar", "humidity_pct", "material_purity", "vibration_mm_s"], 0.05, r)
    df.loc[n:, tname] = np.nan
    return df


def credit(n=1500, new=100, seed=3):
    r = np.random.default_rng(seed); N = n + new
    age = r.normal(41, 11, N).clip(21, 75).round()
    inc = r.lognormal(10.9, 0.45, N).round(-2)
    loan = (inc * r.uniform(0.1, 0.9, N)).round(-2)
    term = r.choice([12, 24, 36, 48, 60], N)
    rate = r.normal(12.5, 3.5, N).clip(4, 28).round(2)
    dti = r.beta(2.2, 5, N).__mul__(60).round(1)
    hist = r.gamma(3.5, 3.0, N).clip(0, 40).round(1)
    delq = r.poisson(0.35, N)
    util = r.beta(2, 3, N).__mul__(100).round(0)
    emp = r.choice(["salaried", "self_employed", "contract", "retired"], N, p=[0.58, 0.2, 0.15, 0.07])
    purpose = r.choice(["debt_consolidation", "home_improvement", "auto", "business", "medical", "other"], N)
    home = r.choice(["rent", "own", "mortgage"], N, p=[0.35, 0.2, 0.45])
    other = r.poisson(1.4, N)
    z = (-3.6 + 0.045 * dti + 0.7 * delq - 0.06 * hist + 0.012 * util + 0.09 * (rate - 12)
         + 0.5 * (emp == "contract") + 0.35 * (emp == "self_employed") + 0.5 * (purpose == "business")
         - 0.4 * (home == "own") + 0.8 * (loan / inc > 0.6) + 0.1 * other + r.normal(0, 0.45, N))
    y = (r.random(N) < _sig(z)).astype(float)
    df = pd.DataFrame(dict(application_id=[f"L{9000 + i}" for i in range(N)], age=age, annual_income=inc,
                           loan_amount=loan, term_months=term, interest_rate=rate, dti_pct=dti,
                           credit_history_years=hist, delinquencies_2y=delq, utilization_pct=util,
                           employment_type=emp, purpose=purpose, home_ownership=home, other_loans=other, defaulted=y))
    df = _holes(df, ["credit_history_years", "utilization_pct", "dti_pct", "employment_type"], 0.04, r)
    df.loc[n:, "defaulted"] = np.nan
    return df


def make(usecase, variant=None):
    if usecase == "clinical": return clinical(), "clinical_readmission_SYNTHETIC.csv"
    if usecase == "credit": return credit(), "loan_portfolio_SYNTHETIC.csv"
    if variant == "failure": return industrial("failure"), "machine_failure_SYNTHETIC.csv"
    return industrial("yield"), "batch_yield_SYNTHETIC.csv"
