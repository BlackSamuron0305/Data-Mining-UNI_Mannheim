# Data-Mining-UNI_Mannheim

IE500 Data Mining, University of Mannheim. Team 5: Predicting Blunders in Online Chess.

## Layout

- `IE500_DM_StudentProject.pdf`: the course's project description
- `submission/`: proposal, final paper and presentation (LaTeX)
- `src/`: the code, with its settings in `config.yaml`
  - `src/preprocessing/selection/`: the data selection
  - `src/preprocessing/bucket_rates/`: blunder probability per bucket of the month (the baseline) and the check of the selection
  - `src/preprocessing/features/`: one row per move with what is known before it is played, and the Stockfish features
  - `src/training/<method>/`: one folder per training method (`bucket_rate`, `logistic_regression`, `gradient_boosting`); `src/training/common.py` holds what they share
  - `src/validation/`: scores every training run on the same moves
- `data/`: the selected games, built by the code and not tracked by git (see `data/README.md`)
- `cache/`: working files of the code, not tracked by git

## Setup

We use [uv](https://docs.astral.sh/uv/) as the package manager, not pip. In the repository run

    uv sync

once. It installs Python 3.14 and the exact package versions of `uv.lock` into `.venv`. Run every script with `uv run`. Add a package with `uv add <name>`. `pyproject.toml`, `uv.lock` and `.python-version` are tracked by git; `.venv` is not.

Copy `.env.example` to `.env` and fill in the keys of the Lichess storage. For the engine features, install [Stockfish](https://stockfishchess.org/download/) and set `STOCKFISH_PATH` in `.env` to the program file.

## Data selection

    uv run python -m src.preprocessing.selection.census                  # counts every game of September 2026 once (about 20 minutes)
    uv run python -m src.preprocessing.selection.stratified_selection    # draws the 350,000 games into data/stratified_selection/ (about 6 minutes)
    uv run python -m src.preprocessing.selection.probabilities           # adds every game's chance to be selected and its weight

## Bucket rates

    uv run python -m src.preprocessing.bucket_rates.rates                # blunder probability of every bucket of the month, into data/bucket_rates/
    uv run python -m src.preprocessing.bucket_rates.selection_check      # the selection against the month, bucket by bucket
    uv run python -m src.preprocessing.bucket_rates.plots                # figures and the table of the results, into data/bucket_rates/figures/

## Features

    uv run python -m src.preprocessing.features.positions                # one row per move of the selected games, into data/features/positions/
    uv run python -m src.preprocessing.features.engine                   # Stockfish on the position before every move of a random subset of games (config.yaml, engine:)

Every feature is known before the move is played: the position, the clocks, the ratings and the earlier moves. The engine ranks the moves the player can choose from (best 5, worst 5, share of losing moves), never the move that was played. A stopped engine run continues where it was.

## Training

Every method has its own folder in `src/training/`, every run writes `data/runs/<method>/<name>/` (predictions of the held-out training games and of the test games, settings, model). Feature sets: `basic` (speed, Elo, clock), `context` (all features without the engine), `engine` (with the engine features; only the analysed games).

    uv run python -m src.training.bucket_rate.train --source month               # baseline: the blunder probability of the move's bucket in the month
    uv run python -m src.training.bucket_rate.train --source train               # the same from the training games only
    uv run python -m src.training.naive_bayes.train --features context            # lower reference among the learners
    uv run python -m src.training.logistic_regression.train --features context
    uv run python -m src.training.decision_tree.train --features context          # symbolic: tunes the depth, writes rules.txt
    uv run python -m src.training.random_forest.train --features context
    uv run python -m src.training.gradient_boosting.train --features context
    uv run python -m src.training.gradient_boosting.train --features engine
    uv run python -m src.training.neural_network.train --features context         # early stopping on the held-out games

Hyperparameters are command-line options (`--help` of each method); give every setting its own `--name`, e.g. `--C 0.1 --name context_C0.1`, and choose between them with the validation below. Why each method is there: Chapter 4 of the final paper.

`--max-games N` trains on a random subset of the training games for a quick run. A new method goes into a new folder `src/training/<method>/train.py` and saves its run with `save_run` of `src/training/common.py`.

## Validation

    uv run python -m src.validation.validate                             # scores every run in data/runs/ on the same moves, into data/validation/

Choose between runs on the held-out training games (valid), report the test games. Runs are compared on the moves that all of them predicted, against the baseline `bucket_rate/month`.
