# System Capacity & Care Load Analytics - Unaccompanied Children (UAC) Program

Only input needed: `HHS_Unaccompanied_Alien_Children_Program.csv` (already included in this folder).
The code finds the CSV automatically, creates every folder it needs, and works from any working directory.

## Files
- `UAC_Care_Load_Analytics.ipynb`  full analysis as a notebook (Colab / Jupyter / VS Code)
- `uac_analysis.py`                the same code as a script (VS Code cells with `# %%`, or terminal)
- `app.py`                         Streamlit dashboard (also written automatically by the notebook)
- `reports/`                       Research paper + Executive summary (.docx / .pdf), `results.json`
- Created on first run: `data/raw/`, `data/processed/`, `figures/`, `reports/results.json`

## Run it
**Google Colab:** upload the .ipynb -> Runtime > Run all. If the CSV is not found, an upload box appears; choose the CSV.
**Jupyter:** open the .ipynb in this folder -> Run All.
**VS Code:** open this folder; either open the .ipynb and Run All, or run `python uac_analysis.py` in a terminal
(or open `uac_analysis.py` and use "Run Cell" on the `# %%` blocks).

## Dashboard
```
pip install -r requirements.txt
streamlit run app.py        # run the analysis once first, it creates data/processed/
```
In Colab, set `RUN_DASHBOARD_IN_COLAB = True` in the last notebook cell (not tested in Colab; locally it is tested).
