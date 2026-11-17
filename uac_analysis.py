# %% [markdown]
# # System Capacity & Care Load Analytics: Unaccompanied Children (UAC) Program
# Runs unchanged in **Google Colab, Jupyter, VS Code (cells or terminal)**. Run the cells top to bottom.
#
# **Only one input is needed: the CSV `HHS_Unaccompanied_Alien_Children_Program.csv`.**
# The notebook looks for it automatically (this folder, `data/raw/`, Downloads, `/content`). In Colab it
# opens an upload box if it cannot find it. To point to it manually, set `CSV_PATH` in the "Locate data" cell.
#
# Outputs are created next to this file: `data/processed/`, `figures/`, `reports/results.json` and `app.py` (dashboard).

# %% 1. Setup (installs missing libraries, detects environment)
import sys, subprocess, importlib, json, shutil, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)


def _ensure(pairs):
    for mod, pip_name in pairs:
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"Installing {pip_name} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


_ensure([("pandas", "pandas"), ("numpy", "numpy"), ("matplotlib", "matplotlib"), ("seaborn", "seaborn")])

import numpy as np
import pandas as pd
import matplotlib

try:
    get_ipython()            # defined in Jupyter / Colab / VS Code interactive
    IN_NB = True
except NameError:
    IN_NB = False
if not IN_NB:
    matplotlib.use("Agg")    # plain script: save figures, do not open windows
import matplotlib.pyplot as plt
import seaborn as sns

IN_COLAB = "google.colab" in sys.modules
try:
    BASE = Path(__file__).resolve().parent     # running as a .py file
except NameError:
    BASE = Path.cwd()                          # running as a notebook / interactive cells
PROC, FIG, REP = BASE / "data" / "processed", BASE / "figures", BASE / "reports"
for d in (BASE / "data" / "raw", PROC, FIG, REP):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
print("Project folder:", BASE, "| notebook:", IN_NB, "| colab:", IN_COLAB)
print("pandas", pd.__version__, "| numpy", np.__version__)


def show(obj):
    """Pretty table in notebooks, plain text in a terminal."""
    if IN_NB:
        from IPython.display import display
        display(obj)
    else:
        print(obj.to_string() if hasattr(obj, "to_string") else obj)


# %% 2. Locate data
CSV_PATH = None   # optional, e.g. r"C:\Users\you\Downloads\HHS_Unaccompanied_Alien_Children_Program.csv"


def _is_uac_csv(p):
    try:
        with open(p, encoding="utf-8-sig") as f:
            return "Children in CBP custody" in f.readline()
    except Exception:
        return False


def find_csv():
    if CSV_PATH:
        p = Path(CSV_PATH).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"CSV_PATH does not exist: {p}")
    folders = [BASE / "data" / "raw", BASE, Path.cwd() / "data" / "raw", Path.cwd(), Path("/content"),
               Path("/content/data/raw"), Path("/mnt/user-data/uploads"), Path.home() / "Downloads"]
    seen = set()
    for d in folders:
        if d in seen or not d.is_dir():
            continue
        seen.add(d)
        for p in sorted(d.glob("*.csv")):
            if _is_uac_csv(p):
                return p
    return None


src = find_csv()
if src is None and IN_COLAB:
    from google.colab import files
    print("CSV not found. Please choose HHS_Unaccompanied_Alien_Children_Program.csv in the upload box:")
    files.upload()                 # Colab saves uploads into the current folder
    src = find_csv()
if src is None:
    raise FileNotFoundError(
        "Could not find the UAC CSV (the file whose header contains 'Children in CBP custody').\n"
        f"Put it in {BASE} or {BASE / 'data' / 'raw'}, or set CSV_PATH in this cell.")
RAW = BASE / "data" / "raw" / src.name
if src.resolve() != RAW.resolve():
    shutil.copy2(src, RAW)         # raw file is only copied, never modified
print("Using raw data:", RAW)

# %% 3. Ingestion and structuring
COLS = ["date", "apprehended", "cbp_custody", "transferred", "hhs_care", "discharged"]
raw = pd.read_csv(RAW, dtype=str, encoding="utf-8-sig")
assert raw.shape[1] == 6, f"Expected 6 columns, found {raw.shape[1]}: {list(raw.columns)}"
raw.columns = COLS                                    # header has an asterisk; rename by position
df = raw.copy()
df["date"] = pd.to_datetime(df["date"].str.strip(), format="%B %d, %Y")
for c in COLS[1:]:
    df[c] = df[c].str.strip().str.replace(",", "", regex=False).astype(int)   # HHS care arrives as "2,484"
df = df.sort_values("date").reset_index(drop=True)    # source file is newest-first
assert df.date.is_unique and int(df.isna().sum().sum()) == 0
print(f"{len(df)} rows, {df.date.min().date()} to {df.date.max().date()}")
show(df.head())

# %% 4. Validation (missing and duplicate dates, logical constraints, anomaly flags)
R = {}                                                # results store, saved to reports/results.json
full_idx = pd.date_range(df.date.min(), df.date.max(), freq="D")
missing = full_idx.difference(df.date)
R["rows_reported"], R["calendar_days"], R["missing_days"] = len(df), len(full_idx), len(missing)
R["duplicate_dates"] = int(df.date.duplicated().sum())
R["missing_weekday"] = {k: int(v) for k, v in pd.Series(missing).dt.day_name().value_counts().items()}
R["reported_weekday"] = {k: int(v) for k, v in df.date.dt.day_name().value_counts().items()}
R["missing_non_fri_sat"] = int(sum(d.day_name() not in ("Friday", "Saturday") for d in missing))
df["flag_transfer_gt_custody"] = df.transferred > df.cbp_custody
df["flag_discharge_gt_hhs"] = df.discharged > df.hhs_care
df["days_since_prev"] = df.date.diff().dt.days
df["flag_gap_before"] = df.days_since_prev > 3        # Thu->Sun (3 days) is the normal Fri+Sat gap
R["flag_transfer_gt_custody"] = int(df.flag_transfer_gt_custody.sum())
R["flag_discharge_gt_hhs"] = int(df.flag_discharge_gt_hhs.sum())
R["flag_gap_before"] = int(df.flag_gap_before.sum())
print({k: R[k] for k in ["rows_reported", "calendar_days", "missing_days", "duplicate_dates",
                         "missing_non_fri_sat", "flag_transfer_gt_custody", "flag_discharge_gt_hhs", "flag_gap_before"]})
print("Missing dates by weekday:", R["missing_weekday"])

# %% 5. Derived metrics (reported days only, nothing imputed)
df["total_load"] = df.cbp_custody + df.hhs_care
df["net_intake"] = df.transferred - df.discharged
df["cum_net_intake"] = df.net_intake.cumsum()
df["growth_rate_pct"] = df.total_load.pct_change() * 100
df["hhs_growth_pct"] = df.hhs_care.pct_change() * 100
x = df.set_index("date")
for w in (7, 14):                                      # calendar-day windows over reported observations
    x[f"load_ma{w}"] = x.total_load.rolling(f"{w}D", min_periods=3).mean()
    x[f"load_sd{w}"] = x.total_load.rolling(f"{w}D", min_periods=3).std()
    x[f"net_ma{w}"] = x.net_intake.rolling(f"{w}D", min_periods=3).mean()
x["local_cv14"] = x.load_sd14 / x.load_ma14 * 100
x["backlog_pos"] = x.net_ma7 > 0


def runs(flag, min_len):
    """True for observations inside runs of >= min_len consecutive True values."""
    g = (flag != flag.shift()).cumsum()
    out = pd.Series(False, index=flag.index)
    for _, s in flag.groupby(g):
        if s.iloc[0] and len(s) >= min_len:
            out.loc[s.index] = True
    return out


x["backlog_sustained"] = runs(x.backlog_pos, 14)       # 7-day avg net intake > 0 for >= 14 reported days
p75, p25 = x.total_load.quantile(.75), x.total_load.quantile(.25)
x["strain"] = runs(x.total_load >= p75, 14)
x["relief"] = runs(x.total_load <= p25, 14)
R["p75_load"], R["p25_load"] = float(p75), float(p25)
df = x.reset_index()
df["year"] = df.date.dt.year
df["month"] = df.date.dt.to_period("M").astype(str)
print(f"Strain threshold (P75) = {p75:,.0f}; relief threshold (P25) = {p25:,.0f}")
show(df[["date", "total_load", "net_intake", "growth_rate_pct", "load_ma7", "load_ma14", "local_cv14"]].tail())

# %% 6. Strain, relief and backlog windows
def spells(col):
    g = (df[col] != df[col].shift()).cumsum()
    rows = []
    for _, s in df.groupby(g):
        if s[col].iloc[0]:
            rows.append(dict(start=str(s.date.iloc[0].date()), end=str(s.date.iloc[-1].date()),
                             reported_days=len(s), calendar_days=(s.date.iloc[-1] - s.date.iloc[0]).days + 1,
                             peak_load=int(s.total_load.max()), mean_load=int(round(s.total_load.mean())),
                             mean_net_intake=round(float(s.net_intake.mean()), 1)))
    return pd.DataFrame(rows)


strain_sp, relief_sp, backlog_sp = spells("strain"), spells("relief"), spells("backlog_sustained")
strain_sp.to_csv(PROC / "strain_windows.csv", index=False)
relief_sp.to_csv(PROC / "relief_windows.csv", index=False)
backlog_sp.to_csv(PROC / "backlog_spells.csv", index=False)
R["strain_windows"], R["relief_windows"] = strain_sp.to_dict("records"), relief_sp.to_dict("records")
R["backlog_spells"] = backlog_sp.to_dict("records")
print("STRAIN windows"); show(strain_sp)
print("RELIEF windows"); show(relief_sp)
print("SUSTAINED BACKLOG spells"); show(backlog_sp)

# %% 7. KPIs
def kpis(d):
    tr, di = d.transferred.sum(), d.discharged.sum()
    bs = d[d.backlog_sustained]
    return dict(
        total_under_care_latest=int(d.total_load.iloc[-1]),
        total_under_care_mean=float(d.total_load.mean()),
        total_under_care_peak=int(d.total_load.max()),
        net_intake_pressure=float(d.net_intake.mean()),
        volatility_index=float(d.local_cv14.mean()),
        backlog_accum_rate=float(bs.net_intake.mean()) if len(bs) else 0.0,
        backlog_time_share_pct=float(len(bs) / len(d) * 100),
        discharge_offset_ratio=float(di / tr))


R["kpi_full"] = kpis(df)
show(pd.Series(R["kpi_full"]).round(2).to_frame("value"))

# %% 8. Temporal and comparative analysis (daily / weekly / monthly, yearly, early vs late)
agg = dict(total_load="mean", cbp_custody="mean", hhs_care="mean", apprehended="mean",
           transferred="mean", discharged="mean", net_intake="mean")
monthly = df.groupby("month").agg(agg).round(1)
weekly = df.set_index("date").resample("W").agg(agg).dropna().round(1)
by_year = df.groupby("year").agg(agg).round(1)
monthly.to_csv(PROC / "monthly_summary.csv")
weekly.to_csv(PROC / "weekly_summary.csv")
R["by_year"] = {int(k): {a: float(b) for a, b in v.items()} for k, v in by_year.to_dict("index").items()}
R["kpi_by_year"] = {int(y): kpis(g) for y, g in df.groupby("year")}
third = len(df) // 3
early, late = df.iloc[:third], df.iloc[-third:]
R["early_late"] = dict(early_range=[str(early.date.min().date()), str(early.date.max().date())],
                       late_range=[str(late.date.min().date()), str(late.date.max().date())],
                       early=kpis(early), late=kpis(late),
                       early_mean_hhs=float(early.hhs_care.mean()), late_mean_hhs=float(late.hhs_care.mean()),
                       early_mean_cbp=float(early.cbp_custody.mean()), late_mean_cbp=float(late.cbp_custody.mean()))
pk, mn = df.loc[df.total_load.idxmax()], df.loc[df.total_load.idxmin()]
R["peak"] = dict(date=str(pk.date.date()), total=int(pk.total_load), hhs=int(pk.hhs_care), cbp=int(pk.cbp_custody))
R["trough"] = dict(date=str(mn.date.date()), total=int(mn.total_load), hhs=int(mn.hhs_care))
R["share_hhs_of_total_pct"] = float(df.hhs_care.sum() / df.total_load.sum() * 100)
R["growth"] = dict(mean=float(df.growth_rate_pct.mean()), sd=float(df.growth_rate_pct.std()),
                   max=float(df.growth_rate_pct.max()), min=float(df.growth_rate_pct.min()),
                   pct_abs_gt3=float((df.growth_rate_pct.abs() > 3).mean() * 100))
R["cv_overall_pct"] = float(df.total_load.std() / df.total_load.mean() * 100)
R["net_intake_stats"] = dict(mean=float(df.net_intake.mean()), pos_share=float((df.net_intake > 0).mean() * 100))
R["peak_apprehended"] = dict(date=str(df.loc[df.apprehended.idxmax(), "date"].date()), v=int(df.apprehended.max()))
R["share_strain_days_pct"] = float(df.strain.mean() * 100)
R["share_relief_days_pct"] = float(df.relief.mean() * 100)
R["yearly_strain_days"] = {int(k): int(v) for k, v in df.groupby("year").strain.sum().items()}
R["yearly_relief_days"] = {int(k): int(v) for k, v in df.groupby("year").relief.sum().items()}
R["corr"] = df[["apprehended", "cbp_custody", "transferred", "hhs_care", "discharged", "net_intake"]].corr().round(2).to_dict()
print("Yearly means"); show(by_year)
print("Monthly means (first 6 rows)"); show(monthly.head(6))

# %% 9. Stock-flow reconciliation (does net intake explain the change in HHS care?)
cons = df[df.days_since_prev == 1].copy()                          # consecutive reported days only
cons["d_hhs"] = cons.hhs_care - df.hhs_care.shift(1).loc[cons.index]
cons["implied_inflow"] = cons.d_hhs + cons.discharged             # inflow needed to explain the HHS change
cons["unexplained"] = cons.implied_inflow - cons.transferred
prev = df.shift(1).loc[cons.index]
R["recon"] = dict(hhs_delta_corr=float(np.corrcoef(cons.d_hhs, cons.net_intake)[0, 1]),
                  cbp_delta_corr=float(np.corrcoef(cons.cbp_custody - prev.cbp_custody,
                                                   cons.apprehended - cons.transferred)[0, 1]))
yr = cons.groupby(cons.date.dt.year)
R["reconciliation"] = dict(
    n_pairs=len(cons), mean_transfers=float(cons.transferred.mean()),
    mean_implied_inflow=float(cons.implied_inflow.mean()), mean_unexplained=float(cons.unexplained.mean()),
    by_year={int(k): round(float(v), 1) for k, v in yr.unexplained.mean().items()},
    implied_by_year={int(k): round(float(v), 1) for k, v in yr.implied_inflow.mean().items()},
    transfers_by_year={int(k): round(float(v), 1) for k, v in yr.transferred.mean().items()},
    total_hhs_change=int(df.hhs_care.iloc[-1] - df.hhs_care.iloc[0]), cum_net_intake=int(df.cum_net_intake.iloc[-1]))
print(json.dumps(R["reconciliation"], indent=1))

# %% 10. Save clean datasets and run validation checks
full = pd.DataFrame({"date": full_idx}).merge(df, on="date", how="left")
full["is_reported"] = full.total_load.notna()
full["weekday"] = full.date.dt.day_name()
full.to_csv(PROC / "uac_full_daily_index.csv", index=False)        # complete daily index, gaps left empty
df.to_csv(PROC / "uac_clean_reported_days.csv", index=False)       # analytic dataset (reported days)
assert len(full) == R["calendar_days"] and int(full.is_reported.sum()) == R["rows_reported"]
assert (df.total_load == df.cbp_custody + df.hhs_care).all()
assert (df.net_intake == df.transferred - df.discharged).all()
assert df.date.is_monotonic_increasing and df.date.is_unique
print("All validation checks passed. Saved to", PROC)

# %% 11. Figures
C = dict(hhs="#1f6f8b", cbp="#e07a1f", tot="#222222", s="#c0392b", r="#2e8b57")


def shade(ax, col, color, label):
    first = True
    for _, s in df.groupby((df[col] != df[col].shift()).cumsum()):
        if s[col].iloc[0]:
            ax.axvspan(s.date.iloc[0], s.date.iloc[-1], color=color, alpha=.15, label=label if first else None)
            first = False


def finish(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150)
    plt.show() if IN_NB else plt.close()


fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(df.date, df.total_load, color="#999", lw=.7, label="Daily total load")
ax.plot(df.date, df.load_ma7, color=C["hhs"], lw=1.6, label="7-day avg")
ax.plot(df.date, df.load_ma14, color=C["cbp"], lw=1.6, label="14-day avg")
shade(ax, "strain", C["s"], f"Strain window (>= P75 = {p75:,.0f})")
shade(ax, "relief", C["r"], f"Relief window (<= P25 = {p25:,.0f})")
ax.set(title="Total System Load (CBP custody + HHS care)", xlabel="Date", ylabel="Children")
ax.legend(loc="upper right", fontsize=8)
finish("fig1_total_load")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax2 = ax.twinx()
ax.plot(df.date, df.hhs_care, color=C["hhs"], lw=1.4, label="HHS care (left axis)")
ax2.plot(df.date, df.cbp_custody, color=C["cbp"], lw=.9, alpha=.8, label="CBP custody (right axis)")
ax2.grid(False)
ax.set(title="CBP custody vs HHS care (separate scales)", xlabel="Date", ylabel="Children in HHS care")
ax2.set_ylabel("Children in CBP custody")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right")
finish("fig2_cbp_vs_hhs")

fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axs[0].plot(df.date, df.net_intake, color="#bbb", lw=.6, label="Daily net intake")
axs[0].plot(df.date, df.net_ma7, color=C["hhs"], lw=1.4, label="7-day avg")
axs[0].plot(df.date, df.net_ma14, color=C["cbp"], lw=1.4, label="14-day avg")
axs[0].axhline(0, color="k", lw=.8)
shade(axs[0], "backlog_sustained", C["s"], "Sustained backlog (>= 14 obs. of positive 7-day avg)")
axs[0].set(title="Net Daily Intake (transfers into HHS - discharges)", ylabel="Children/day")
axs[0].legend(fontsize=8, loc="upper right")
axs[1].plot(df.date, df.cum_net_intake, color=C["tot"])
axs[1].set(title="Cumulative net intake (reported days only)", ylabel="Children", xlabel="Date")
finish("fig3_net_intake_backlog")

fig, ax = plt.subplots(figsize=(11, 4.5))
m = monthly.copy()
m.index = pd.to_datetime(m.index)
ax.bar(m.index, m.hhs_care, width=22, color=C["hhs"], label="HHS care (monthly mean)")
ax.bar(m.index, m.cbp_custody, width=22, bottom=m.hhs_care, color=C["cbp"], label="CBP custody (monthly mean)")
ax.set(title="Monthly average care load (stacked)", ylabel="Children", xlabel="Month")
ax.legend()
finish("fig4_monthly")

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
yy = pd.DataFrame(R["kpi_by_year"]).T
axs[0].bar(by_year.index.astype(str), by_year.total_load, color=C["hhs"])
axs[0].set_title("Mean total load")
axs[1].bar(by_year.index.astype(str), by_year.net_intake, color=[C["r"] if v < 0 else C["s"] for v in by_year.net_intake])
axs[1].axhline(0, color="k", lw=.8)
axs[1].set_title("Mean net daily intake")
axs[2].bar(yy.index.astype(str), yy.discharge_offset_ratio, color=C["cbp"])
axs[2].axhline(1, color="k", lw=.8, ls="--")
axs[2].set_title("Discharge offset ratio")
fig.suptitle("Calendar-year comparison")
finish("fig5_year_compare")

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(df[["apprehended", "cbp_custody", "transferred", "hhs_care", "discharged", "net_intake"]].corr(),
            annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation between pipeline stages")
finish("fig6_corr")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(df.date, df.local_cv14, color=C["tot"], lw=1)
axs[0].set(title="Care Load Volatility (14-day rolling CV, %)", xlabel="Date", ylabel="%")
axs[1].hist(df.growth_rate_pct.dropna(), bins=50, color=C["hhs"])
axs[1].set(title="Day-over-day load growth (%)", xlabel="% change", ylabel="Days")
finish("fig7_volatility_growth")

order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
wd = pd.DataFrame({"Reported": df.date.dt.day_name().value_counts(),
                   "Missing": pd.Series(missing).dt.day_name().value_counts()}).fillna(0).reindex(order)
fig, ax = plt.subplots(figsize=(8, 4))
wd.plot(kind="bar", stacked=True, ax=ax, color=[C["hhs"], "#bbb"])
ax.set(title="Reporting coverage by weekday", ylabel="Calendar days", xlabel="")
plt.xticks(rotation=0)
finish("fig8_coverage")

# %% 12. Save results and print summary
def _json(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


with open(REP / "results.json", "w") as f:
    json.dump(R, f, indent=1, default=_json)
print("Saved:", REP / "results.json")
print(f"Peak total load {R['peak']['total']:,} on {R['peak']['date']}; latest {R['kpi_full']['total_under_care_latest']:,}")
print(f"HHS share of load {R['share_hhs_of_total_pct']:.1f}%; discharge offset ratio {R['kpi_full']['discharge_offset_ratio']:.2f}")
print("Figures saved in:", FIG)
