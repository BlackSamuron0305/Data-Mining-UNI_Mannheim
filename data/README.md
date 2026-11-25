# data

Everything here is built by the scripts in `src/` (or downloaded from our server, as `data:` in `config.yaml` says)
and is not tracked by git, apart from this file. There is one folder per step, named like the folders of `src/`, and
inside it one folder per month or selection. `work/` holds the files in between and `logs/` the logs; neither goes
to the server.

    data/
    ├── 00_raw/<month>/                 the raw Lichess file of the month (about 30 GB), only for start_with: raw
    ├── 01_selection/
    │   ├── september/                  the count of the month and the blunder probability of every bucket
    │   ├── august/                     the same for August, only to compare
    │   ├── random/                     a random selection, only to compare
    │   ├── stratified/                 the stratified run: 350,000 games, training games as drawn, test balanced
    │   ├── stratified_balanced/        the balanced run: the same test games, training games balanced too
    │   ├── comparison/                 the selections against the month, and the two months against each other
    │   └── work/                       the pool of games the selections draw from
    ├── 02_preprocessing/
    │   ├── stratified/                 one row per move of the stratified run, the columns we keep
    │   ├── stratified_balanced/        the same for the balanced run
    │   ├── stockfish/                  the Stockfish columns of every analysed move
    │   └── work/                       the moves before cleaning, and Stockfish's batches while it runs
    ├── 03_transformation/
    │   ├── stratified/                 the tables the models read
    │   └── stratified_balanced/
    ├── 04_data_mining/
    │   └── <selection>/<method>/<run>/ the training runs
    └── 05_evaluation/
        ├── <selection>/                the scores of every run
        └── before_reduction/           the scores of the first version, before the column reduction

`<month>` is `september` (the month we use) or `august` (only counted, to compare its blunder probability).
`<selection>` is `stratified` or `stratified_balanced`, set by `IE500_SELECTION` (default `stratified`).

## 01_selection

- `00_raw/<month>/` comes from `src/01_selection/download.py`.
- `<month>/` comes from `cleanup.py`: `cleanup.json`, `funnel.csv` (games after each filter step), `groups.csv` and
  `buckets.csv` (counts of the month); and from `rates.py`: `rates.csv` (blunder probability of every bucket, with
  its 95% interval and the probability the baseline predicts), `by_time_left.csv`, `summary.md`, and `figures/` from
  `rates_plots.ipynb`. The pool (`pool.parquet`, `pool_buckets.parquet`) goes to `work/`.
- `<selection>/` comes from `select.py --run <selection>`:

      summary.md            the main numbers; start here
      selection.json        settings, balancing statistics, totals
      train.pgn.zst         training games in their original PGN text
      test.pgn.zst          test games in their original PGN text
      games.csv             one row per game, with its chance to be picked and its weight
      groups.csv            games per group (speed x Elo of the mean rating), with its chance and weight
      buckets.csv           moves and blunders per bucket (speed x Elo of the mover x time left)

- `comparison/` comes from `compare.py`: `summary.md`, `comparison.json`, `buckets.csv` (rates and z per bucket of
  every selection), `by_time_left.csv`, `weights.csv`, and `figures/` from `compare_report.ipynb`; and from
  `months.py`: `months.csv` and `months.md` (blunder probability of September and August).

## 02_preprocessing

- `<selection>/` comes from `src/02_preprocessing/`: `train.parquet` and `test.parquet` (one row per move, what step 3
  reads), `removed.csv` (every removed column, its rule and the measured numbers), `kept.csv` (the columns kept
  whatever the numbers say), `false_predictors.csv`, `extract.json`. The stratified run also has `leakage_check.json`
  and `inspection/` (columns with their information about blunders, correlations, `figures/` from `report.ipynb`);
  the balanced run takes its decisions.
- The tables hold the Stockfish columns (`sf_*`). A model without Stockfish leaves them out; the reduction makes sure
  that this is exactly the table without Stockfish.
- `stockfish/` comes from `stockfish.py` or our server: `train.parquet`, `test.parquet` (one row per analysed move of
  every selection) and `engine.json` (engine and settings).
- `work/<selection>/` holds `extracted_<part>.parquet` (every column, incl. the false predictors) and
  `clean_<part>.parquet` (false predictors removed, Stockfish columns added); `work/stockfish/` the batches.

## 03_transformation

`<selection>/` comes from `src/03_transformation/transform.py`: `train.parquet` and `test.parquet` (the final tables of
step 2 with fixed categories, 1/0 for true/false, 32-bit floats and, for the training games, the column `fold`),
`transform.json`.

## On our server

The server holds a copy of `01_selection/`, `02_preprocessing/` and `03_transformation/` without `work/` and `logs/`
(`uv run python -m src.common.server list`). A later step that needs a small file of an earlier one, such as
`01_selection/september/rates.csv`, downloads just that file.

## 04_data_mining

`<selection>/<method>/<name>/` comes from `src/04_data_mining/<method>/train.py`: `run.json`,
`predictions_valid.parquet`, `predictions_test.parquet`, `model.joblib`; tuned runs (`<features>_tuned`, from
`src/04_data_mining/tune.py`) add `tuning.csv`, and `src/04_data_mining/importance.py` adds `importance_groups.csv`.
The ablation writes `stratified/ablation/<method>_<features>_<variant>/`; the logs of `run_models.sh` go to
`<selection>/logs/`.

## 05_evaluation

`<selection>/` comes from `src/05_evaluation/validate.py`: `summary.md`, `results.csv` (all scores, incl. those at each
run's cut-off and among the 5% riskiest moves), `intervals.csv` (95% intervals, resampling whole games),
`by_speed.csv`, `by_subgroup.csv`, `calibration.csv`, `figures/` (ROC and precision-recall, calibration, gains,
subgroups) and `errors/` (error analysis of the best run). With `--with stratified_balanced` the same goes to
`stratified/with_stratified_balanced/`.
