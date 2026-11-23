"""Generate the report's LaTeX tables from the Part 2 results (never type numbers by hand).

Run from anywhere:  python 3_report/build_tables.py
Writes 3_report/tables/*.tex (booktabs). Re-run after re-running the Part 2 notebook.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "2_model" / "results"
FLOW = ROOT / "data" / "processed" / "cohort_flow.csv"
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)

NAMES = {"LogReg": "Logistic regression", "RandomForest": "Random forest", "XGBoost": "XGBoost",
         "LightGBM": "LightGBM", "MLP": "MLP", "SOFA": "SOFA score", "Ensemble": "Ensemble (final)"}


def tex(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("–", "--").replace("−", "$-$"))


def rnd(v, d=3, sign=False):
    # round half away from zero (0.0625 -> 0.063), unlike binary float formatting
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{q:+.{d}f}" if sign else f"{q:.{d}f}"


def ci(v, lo, hi, d=3):
    return f"{rnd(v, d)} ({rnd(lo, d)}--{rnd(hi, d)})"


def write(name, body):
    (OUT / f"{name}.tex").write_text(body.strip() + "\n", encoding="utf-8")
    print("wrote", name)


# ---------------------------------------------------------------- cohort and split
rows = []
if FLOW.exists():
    flow = pd.read_csv(FLOW)
    for _, r in flow.iterrows():
        excluded = "--" if r["excluded"] == 0 else f"{int(r['excluded']):,}"
        step = tex(r["step"]).replace(">=", r"$\geq$")
        rows.append(f"{step} & {int(r['stays']):,} & {excluded} \\\\")
split = pd.read_csv(RES / "split_summary.csv")
write("tab_cohort", r"""
\begin{tabular}{@{}lrr@{}}
\toprule
Inclusion step & Stays & Excluded \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
Split (patient-grouped) & Stays & Deaths \ (rate) \\
\midrule
""" + "\n".join(f"{tex(r['set'])} & {r['stays']:,} & {r['deaths']:,} ({r['mortality_%']:.2f}\\%) \\\\" for _, r in split.iterrows()) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- main test results (table*)
t = pd.read_csv(RES / "test_model_comparison.csv").set_index("model")
order = [m for m in ["Ensemble", "XGBoost", "LightGBM", "MLP", "RandomForest", "LogReg", "SOFA"] if m in t.index]
body = []
for m in order:
    r = t.loc[m]
    name = r"\textbf{" + NAMES[m] + "}" if m == "Ensemble" else NAMES[m]
    cells = [ci(r[k], r[f"{k}_lo"], r[f"{k}_hi"]) for k in ["AUROC", "AUPRC", "Brier", "ECE", "Cal_slope", "Sensitivity", "Specificity"]]
    body.append(name + " & " + " & ".join(cells) + r" \\")
write("tab_results", r"""
\begin{tabular}{@{}lccccccc@{}}
\toprule
Model & AUROC & AUPRC & Brier & ECE & Cal.\ slope & Sensitivity & Specificity \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- paired bootstrap
p = pd.read_csv(RES / "test_paired_bootstrap.csv")
p = p[p["metric"].isin(["AUPRC", "ECE"])]
body = [f"{tex(row['comparison'])} & {row['metric']} & ${rnd(row['difference'], sign=True)}$ & "
        f"$[{rnd(row['ci_low'], sign=True)},\\ {rnd(row['ci_high'], sign=True)}]$ \\\\"
        for _, row in p.iterrows()]
write("tab_paired", r"""
\begin{tabular}{@{}llrr@{}}
\toprule
Comparison (A $-$ B) & Metric & $\Delta$ & 95\% CI \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- tuning
tun = pd.read_csv(RES / "cv_default_vs_tuned.csv").set_index("model")
ts = pd.read_csv(RES / "tuning_best_params.csv").set_index("model")
body = [f"{NAMES[m]} & {int(ts.loc[m, 'trials'])} & {tun.loc[m, 'cv_AUPRC_default']:.3f} & {tun.loc[m, 'cv_AUPRC_tuned']:.3f} & ${tun.loc[m, 'AUPRC_gain']:+.3f}$ \\\\"
        for m in tun.index]
write("tab_tuning", r"""
\begin{tabular}{@{}lrrrr@{}}
\toprule
Model & Trials & Default & Tuned & Gain \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- calibration raw vs calibrated
c = pd.read_csv(RES / "test_calibration_raw_vs_calibrated.csv")
body = []
for m in c["model"].unique():
    raw = c[(c["model"] == m) & (c["probabilities"] == "raw")].iloc[0]
    cal = c[(c["model"] == m) & (c["probabilities"] != "raw")].iloc[0]
    body.append(f"{NAMES[m]} & {raw['ECE']:.4f} & {cal['ECE']:.4f} & {raw['slope']:.2f} & {cal['slope']:.2f} \\\\")
write("tab_calibration", r"""
\begin{tabular}{@{}lrrrr@{}}
\toprule
 & \multicolumn{2}{c}{ECE} & \multicolumn{2}{c}{Cal.\ slope} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Model & Raw & Isotonic & Raw & Isotonic \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- missing-data strategies
md = pd.read_csv(RES / "protocol_missing_data_strategies.csv")
allt = md[md["evaluated_on"] == "all test stays"].set_index(["strategy", "model"])
cc = md[md["evaluated_on"] != "all test stays"].set_index(["strategy", "model"])
body = []
for (s, m), r in allt.iterrows():
    body.append(f"{tex(s)} & {NAMES[m]} & {r['train_rows_%']:.0f}\\% & {r['AUROC']:.3f} & {r['AUPRC']:.3f} & {r['ECE']:.3f} & {cc.loc[(s, m), 'AUPRC']:.3f} \\\\")
write("tab_missing", r"""
\begin{tabular}{@{}llrrrrr@{}}
\toprule
 & & & \multicolumn{3}{c}{All test stays} & CC subset \\
\cmidrule(lr){4-6}\cmidrule(lr){7-7}
Strategy & Model & Train rows & AUROC & AUPRC & ECE & AUPRC \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- temporal hold-out
tp = pd.read_csv(RES / "protocol_temporal_holdout.csv")
SHORT = {"Logistic regression": "LR", "XGBoost": "XGBoost"}
def period(s):
    return "2008--2019" if s.startswith("same era") else s.split("(")[-1].rstrip(")").replace(" - ", "--")
body = [f"{SHORT.get(NAMES[row['model']], NAMES[row['model']])} & {period(row['test_period'])} & {row['n']:,} & "
        f"{tex(row['AUROC'])} & {tex(row['AUPRC'])} \\\\" for _, row in tp.iterrows()]
write("tab_temporal", r"""
\begin{tabular}{@{}llrcc@{}}
\toprule
Model & Test period & $n$ & AUROC & AUPRC \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- subgroup gaps
g = pd.read_csv(RES / "test_subgroup_gaps.csv")
LBL = {"age_band": "Age band", "insurance": "Insurance", "anchor_year_group": "Admission period",
       "careunit_group": "Care unit", "race_group": "Race", "gender": "Sex", "admission_group": "Admission type"}
body = [f"{LBL.get(r['attribute'], tex(r['attribute']))} & {int(r['groups'])} & {r['AUROC_min']:.3f}--{r['AUROC_max']:.3f} & {r['AUROC_gap']:.3f} & {r['ECE_max']:.3f} \\\\"
        for _, r in g.iterrows()]
write("tab_subgroups", r"""
\begin{tabular}{@{}lrccc@{}}
\toprule
Attribute & Groups & AUROC range & Gap & Max ECE \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- key numbers for the authors
summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
(OUT / "key_numbers.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print("done")
