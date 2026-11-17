# data

Everything here is built by the scripts in `src/` and is not tracked by git, apart from this file.

- `stratified_selection/` comes from `uv run python -m src.preprocessing.selection.stratified_selection`, then `uv run python -m src.preprocessing.selection.probabilities`. It holds the 350,000 selected games of September 2026:

      stratified_selection/
        summary.md            the main numbers; start here
        selection.json        settings, balancing statistics, totals
        games/                the games themselves
          train.pgn.zst         training games in their original PGN text
          test.pgn.zst          test games in their original PGN text
          games.csv             one row per game, including its chance to be selected and its weight
        tables/               counts of the month next to the selection's
          groups.csv            games per group (speed x Elo of the mean rating), with its chance and weight
          buckets.csv           moves and blunders per bucket (speed x Elo of the mover x time left)
        logs/                 logs of the runs
          selection.log, probabilities.log

- `bucket_rates/` comes from `src/preprocessing/bucket_rates/`: `rates.csv` (blunder probability of every bucket of the month, with its 95% interval and the probability the baseline predicts), `by_time_left.csv`, `selection_check.csv`/`.json`/`.md` (the selection against the month and against random draws), `selection_by_time_left.csv`, `figures/`, `summary.md`.
- `features/` comes from `src/preprocessing/features/`:

      features/
        positions/            one row per move: train.parquet, test.parquet, positions.json
        engine/               Stockfish features per move of the analysed games
          train/, test/         one parquet file per batch of games, engine.json (settings and engine name)

- `runs/<method>/<name>/` comes from `src/training/<method>/train.py`: `run.json`, `predictions_valid.parquet`, `predictions_test.parquet`, `model.joblib`.
- `validation/` comes from `src/validation/validate.py`: `summary.md`, `results.csv`, `by_speed.csv`, `calibration.csv`.

The working files of the selection (the full count of September and the pool of candidate games, about 650 MB) are in `cache/census/`, which is not tracked by git either.
