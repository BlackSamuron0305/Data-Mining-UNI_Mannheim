# data

Everything here is built by the scripts in `src/` (or downloaded from our server, as `data:` in `config.yaml` says)
and is not tracked by git, apart from this file. There is one folder per step, named like the folders of `src/`.

    data/
    ├── 01_selection/
    │   ├── raw/<month>/                      the raw Lichess file of the month (about 30 GB), only for start_with: raw
    │   ├── preparation/
    │   │   ├── cleanup/<month>/              the count of the month and the pool of games to draw from
    │   │   └── blunder_rates/<month>/        blunder probability of every bucket, with figures/
    │   └── sampling/
    │       ├── stratified/                   the stratified run: 350,000 games, training games as drawn, test balanced
    │       ├── stratified_balanced/          the balanced run: the same test games, training games balanced too
    │       └── comparison/                   the selections against the month
    ├── 02_preprocessing/
    │   ├── stratified/                       one row per move, for the stratified run
    │   └── stratified_balanced/              the same for the balanced run
    ├── 03_transformation/
    │   └── <selection>/                      the tables the models read
    ├── 04_data_mining/
    │   └── <selection>/<method>/<run>/       the training runs
    └── 05_evaluation/
        ├── <selection>/                      the scores of every run
        └── before_reduction/                 the scores of the first version, before the column reduction

`<month>` is `september` (the month we use) or `august` (only counted, to compare its blunder probability). `<selection>` is `stratified` or `stratified_balanced`, set by
`IE500_SELECTION` (default `stratified`).

## 01_selection

- `raw/<month>/` comes from `src/01_selection/download/`.
- `preparation/cleanup/<month>/` comes from `src/01_selection/preparation/cleanup/cleanup.py`: `cleanup.json`,
  `funnel.csv` (games after each filter step), `groups.csv` and `buckets.csv` (counts of the month), `pool.parquet` and
  `pool_buckets.parquet` (the games the selections draw from), `cleanup.log`.
- `preparation/blunder_rates/<month>/` comes from `src/01_selection/preparation/blunder_rates/rates.py`: `rates.csv`
  (blunder probability of every bucket, with its 95% interval and the probability the baseline predicts),
  `by_time_left.csv`, `summary.md`, and `figures/` from `plots.ipynb`. `preparation/blunder_rates/months/` comes from
  `months.py`: `months.csv` and `summary.md` (blunder probability of September and August, per speed and in total)
  and `figures/months_by_time_left` from `plots.ipynb`.
- `sampling/<selection>/` comes from `src/01_selection/sampling/<selection>/select.py`:

      summary.md            the main numbers; start here
      selection.json        settings, balancing statistics, totals
      games/
        train.pgn.zst         training games in their original PGN text
        test.pgn.zst          test games in their original PGN text
        games.csv             one row per game, with its chance to be picked and its weight
      tables/
        groups.csv            games per group (speed x Elo of the mean rating), with its chance and weight
        buckets.csv           moves and blunders per bucket (speed x Elo of the mover x time left)
      logs/selection.log

- `sampling/comparison/` comes from `src/01_selection/sampling/comparison/compare.py`: `summary.md`,
  `comparison.json`, `buckets.csv` (rates and z per bucket of every selection), `by_time_left.csv`, `weights.csv`, and
  `figures/` from `report.ipynb`.

## 02_preprocessing

`<selection>/` comes from `src/02_preprocessing/`: `extracted/` (every column, incl. the false predictors),
`<without|with>_stockfish/clean/` (false predictors removed), `inspection/` (columns with their information about
blunders, correlations, figures), `final/` (reduced, with `removed.csv`: every removed column, its rule and the
measured numbers, and `kept.csv`: the columns kept whatever the numbers say; what step 3 reads),
`with_stockfish/stockfish/` (the Stockfish columns,
downloaded or computed; the balanced run reads the stratified run's, as they share their games), `logs/leakage_check.log`.

## 03_transformation

`<selection>/` comes from `src/03_transformation/transform.py`: `<without|with>_stockfish/train.parquet` and
`test.parquet` (the final tables of step 2 with fixed categories, 1/0 for true/false, 32-bit floats and, for the
training games, the column `fold`), `transform.json`, `logs/`.

## What each state on our server holds

`selected`: everything of `01_selection/` except `raw/` and the pool (`pool.parquet`, `pool_buckets.parquet`).
`preprocessed`: everything of `02_preprocessing/` except `extracted/`, `clean/` and `stockfish/`. `transformed`:
`03_transformation/`. The last two also hold what steps 4 and 5 read of step 1 (`rates.csv` of the blunder rates and
`games.csv` of the selections). `stockfish`: the Stockfish columns of both runs, one file per part.

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
