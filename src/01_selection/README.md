# 01 Selection

The raw Lichess month becomes the games we work with: one pass over the month, the blunder probability of every
bucket, and three selections of 350,000 games, each split into training and test games. Four folders, the scripts
in the order they run (`src/pipeline.py` runs them for you):

| folder | script | what it does | writes to `data/` |
|---|---|---|---|
| `raw/` | `download.py` | the raw month (about 30 GB) from Lichess or our server; `storage.py` reads it frame by frame | `00_raw/<month>/` |
| `month/` | `cleanup.py` | one pass over the month: drops the games that are not complete cases (`filters.py`), counts moves and blunders per bucket and group, keeps a tenth of the games as the pool | `01_selection/<month>/`, pool in `01_selection/work/` |
| | `rates.py` | the blunder probability of every bucket with its 95% interval (`intervals.py`); figures: `rates_plots.ipynb` | `01_selection/<month>/` |
| `sampling/` | `select.py --run <selection>` | draws `random`, `stratified` or `stratified_balanced` from the pool: the draw, the split and the weights in `draw.py`, the balancing in `swap.py`, the files in `write.py` | `01_selection/<selection>/` |
| `comparison/` | `compare.py` | the three selections against the month, bucket by bucket; figures: `compare_report.ipynb` | `01_selection/comparison/` |
| | `months.py` | September against August, after the cleanup and the rates of both months (`IE500_MONTH=2026-08`) | `01_selection/comparison/` |

Run one with its folder, e.g. `uv run python -m src.01_selection.sampling.select --run stratified`. All commands:
section "Data selection" of the `README.md` at the root; what every file in `data/` holds: `data/README.md`.
