# data

Everything here is built by the scripts in `src/` and is not tracked by git, apart from this file.

- `selection/<random|stratified|balanced>/` comes from `src/01_selection/sampling/<name>/select.py`. Each holds 350,000 selected games of September 2026:

      selection/stratified/
        summary.md            the main numbers; start here
        selection.json        settings, balancing statistics, totals
        games/                the games themselves
          train.pgn.zst         training games in their original PGN text
          test.pgn.zst          test games in their original PGN text
          games.csv             one row per game, with its chance to be picked and its weight
        tables/               counts of the month next to the selection's
          groups.csv            games per group (speed x Elo of the mean rating), with its chance and weight
          buckets.csv           moves and blunders per bucket (speed x Elo of the mover x time left)
        logs/selection.log

- `selection/comparison/` comes from `src/01_selection/comparison/compare.py`: `summary.md`, `comparison.json`, `buckets.csv` (rates and z per bucket of every selection), `by_time_left.csv`, `weights.csv`, and `figures/` from `report.ipynb`.
- `blunder_rates/` comes from `src/01_selection/preparation/blunder_rates/rates.py`: `rates.csv` (blunder probability of every bucket of the month, with its 95% interval and the probability the baseline predicts), `by_time_left.csv`, `summary.md`, and `figures/` from `plots.ipynb`.
- `preprocessing/` comes from `src/02_preprocessing/`: `extracted/` (every column, incl. the false predictors), `<without|with>_stockfish/clean/` (false predictors removed), `inspection/` (columns, correlations, figures), `final/` (reduced, with `removed.csv`), `with_stockfish/stockfish/` (the Stockfish columns, downloaded or computed), `logs/leakage_check.log`.
- `features/` is the move table of the earlier code that the current training runs read (until the training switches to `preprocessing/*/final/`):

      features/
        positions/            one row per move: train.parquet, test.parquet, positions.json
        engine/               Stockfish features per move (top 5 and worst 5 of the moves the player could choose)
          train/, test/         one parquet file per batch of games, engine.json (settings and engine name)

- `runs/<method>/<name>/` comes from `src/04_data_mining/<method>/train.py`: `run.json`, `predictions_valid.parquet`, `predictions_test.parquet`, `model.joblib`; tuned runs (`<features>_tuned`, from `src/04_data_mining/tune.py`) add `tuning.csv`, and `src/04_data_mining/importance.py` adds `importance_groups.csv`. The ablation writes `runs/ablation/<method>_<features>_<variant>/`.
- `validation/` comes from `src/05_evaluation/validate.py`: `summary.md`, `results.csv` (all scores, incl. those at each run's cut-off and among the 5% riskiest moves), `intervals.csv` (95% intervals, resampling whole games), `by_speed.csv`, `by_subgroup.csv`, `calibration.csv`, `figures/` (ROC and precision-recall, calibration, gains, subgroups) and `errors/` (error analysis of the best run). With `--with LABEL=FOLDER` the same goes to `validation/with_<LABEL>/`.

`data_balanced/` next to `data/` has the same layout for the balancing experiment (README.md), built with `IE500_DATA=data_balanced IE500_SELECTION=balanced`.

The working files of the cleanup (the full count of September and the pool of games, about 650 MB) are in `cache/cleanup/`, which is not tracked by git either. Raw monthly files from the download scripts go to `cache/raw/`.
