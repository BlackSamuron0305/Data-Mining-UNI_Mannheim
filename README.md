# Data-Mining-UNI_Mannheim

IE500 Data Mining, University of Mannheim. Team 5: Predicting Blunders in Online Chess.

## Layout

- `IE500_DM_StudentProject.pdf`: the course's project description
- `submission/`: proposal, final paper and presentation (LaTeX)
- `src/`: the code, with its settings in `config.yaml`, in the five steps of the data mining process (Fayyad et al., 1996)
  - `src/01_selection/`: getting the data and choosing the games
    - `download/`: the raw files of September and August, from our server or from Lichess (`storage.py`)
    - `preparation/cleanup/`: one pass over the month that drops unwanted games, counts moves and blunders per bucket and keeps a pool of games
    - `preparation/blunder_rates/`: the blunder probability per bucket of the month, with `plots.ipynb`
    - `sampling/`: the `random`, `stratified` and `stratified_balanced` selection, each in its own folder, and the code they share in `common/`
    - `comparison/`: the three selections against the month, with `report.ipynb` (incl. the weights)
  - `src/02_preprocessing/`: the selected games turned into one row per move, the false predictors removed and the columns reduced
    - `common/`: the extraction (every column, incl. the false predictors), the list of false predictors, the leakage check, the inspection
    - `without_stockfish/`: preprocess, inspect (`report.ipynb`: correlation heatmap and decisions), reduce
    - `with_stockfish/`: the same with the 15 Stockfish columns; Stockfish comes from the team's server by default (its `README.md`)
  - `src/03_transformation/encoding/`: turns the columns into what the models can use: gaps filled, numbers scaled, categories as 0/1 columns or as numbers for the trees (its `README.md`)
  - `src/04_data_mining/<method>/`: one folder per training method; `src/04_data_mining/common.py` holds what they share
  - `src/05_evaluation/`: scores every training run on the same moves
  - `src/common/`: code used by several steps
    - `pgn.py`: reads a Lichess game (moves, clocks, evaluations, blunder marks); steps 1 and 2
    - `server.py`: lists, downloads and uploads our prepared data on the team's server; steps 1 and 2, and on its own
    - `buckets.py`: the groups of games and buckets of moves; steps 1 to 5
    - `metrics.py`: the scores of the predictions; steps 4 and 5
    - `plot_style.py`: colours and sizes of the figures; the notebooks of steps 1 and 2
- `data/`: the selected games, built by the code and not tracked by git (see `data/README.md`)
- `cache/`: working files of the code, not tracked by git

## Setup

We use [uv](https://docs.astral.sh/uv/) as the package manager, not pip. In the repository run

    uv sync

once. It installs Python 3.14 and the exact package versions of `uv.lock` into `.venv`. Run every script with `uv run`. Add a package with `uv add <name>`. `pyproject.toml`, `uv.lock` and `.python-version` are tracked by git; `.venv` is not.

Copy `.env.example` to `.env`, then choose one of the two ways below with `IE500_SOURCE` there. The way from our server also needs the keys of the team's storage (S3) in `.env`; the way from scratch needs none.

## Two ways to run

Both ways run the same code and give the same tables; they differ only in where the slow results come from. Each step is described in the sections after this one.

| | from our server (`IE500_SOURCE=server`, the default) | from scratch (`IE500_SOURCE=scratch`) |
|---|---|---|
| needs | the keys in `.env` | Stockfish (`STOCKFISH_PATH` in `.env`), about 40 GB of disk |
| raw month | not needed | 30 GB from Lichess |
| cleanup of the month | downloaded (about 650 MB) | computed (about 20 minutes) |
| selection | computed (about 1 minute) | computed (about 1 minute) |
| preprocessing and Stockfish | the final tables downloaded (about 1.7 GB) | computed (Stockfish: about 15 hours on 16 cores) |
| transformation, training, evaluation | computed | computed |

Whatever a step needs and does not find, it downloads from the server, or, with `scratch`, it stops and names the step to run first. Downloads are checked against their sha256 and go to the same folders as the computed files, so the later steps cannot tell the difference.

### From our server

    uv run python -m src.01_selection.preparation.blunder_rates.rates     # downloads the cleanup
    uv run python -m src.01_selection.sampling.stratified.select
    export IE500_DATA=data_balanced IE500_SELECTION=balanced              # run B (balancing experiment)
    uv run python -m src.01_selection.preparation.blunder_rates.rates
    uv run python -m src.01_selection.sampling.stratified_balanced.select
    unset IE500_DATA IE500_SELECTION
    bash run_models.sh                                                    # downloads the final tables, trains, validates

### From scratch

    uv run python -m src.01_selection.download.september                  # 30 GB from Lichess
    uv run python -m src.01_selection.preparation.cleanup.cleanup
    uv run python -m src.01_selection.preparation.blunder_rates.rates
    uv run python -m src.01_selection.sampling.stratified.select
    uv run python -m src.02_preprocessing.common.extract
    uv run python -m src.02_preprocessing.common.leakage_check
    uv run python -m src.02_preprocessing.without_stockfish.preprocess
    uv run python -m src.02_preprocessing.without_stockfish.inspection
    uv run python -m src.02_preprocessing.without_stockfish.reduce
    uv run python -m src.02_preprocessing.with_stockfish.engine           # about 15 hours
    uv run python -m src.02_preprocessing.with_stockfish.preprocess
    uv run python -m src.02_preprocessing.with_stockfish.inspection
    uv run python -m src.02_preprocessing.with_stockfish.reduce
    export IE500_DATA=data_balanced IE500_SELECTION=balanced              # run B (balancing experiment)
    uv run python -m src.01_selection.preparation.blunder_rates.rates
    uv run python -m src.01_selection.sampling.stratified_balanced.select
    uv run python -m src.02_preprocessing.common.extract
    IE500_ENGINE_REUSE=data uv run python -m src.02_preprocessing.with_stockfish.engine   # only run B's new games
    uv run python -m src.02_preprocessing.without_stockfish.preprocess
    uv run python -m src.02_preprocessing.without_stockfish.reduce
    uv run python -m src.02_preprocessing.with_stockfish.preprocess
    uv run python -m src.02_preprocessing.with_stockfish.reduce
    unset IE500_DATA IE500_SELECTION
    bash run_models.sh

Run B uses the column decisions of run A: its `reduce.py` removes the same columns, no new inspection.

`uv run python -m src.common.server list` shows what is on the server; `download` fetches it on its own (`download cleanup`, `download stockfish`, `download preprocessed`). `upload` puts new versions there and needs a key that may write; `upload preprocessed` uploads the final tables of `IE500_SELECTION`.

## Data selection

    uv run python -m src.01_selection.download.september                 # the raw file of September 2026 into cache/raw/ (about 30 GB, checked against Lichess's sha256)
    uv run python -m src.01_selection.download.august                    # the same for August 2026
    uv run python -m src.01_selection.preparation.cleanup.cleanup        # one pass over the month into cache/cleanup/ (about 20 minutes)
    uv run python -m src.01_selection.preparation.blunder_rates.rates    # blunder probability per bucket into data/blunder_rates/; figures: plots.ipynb
    uv run python -m src.01_selection.sampling.random.select             # 350,000 games at random into data/selection/random/
    uv run python -m src.01_selection.sampling.stratified.select         # stratified by speed x Elo into data/selection/stratified/
    uv run python -m src.01_selection.sampling.stratified_balanced.select  # stratified, training games balanced, into data/selection/balanced/
    uv run python -m src.01_selection.comparison.compare                 # the three against the month into data/selection/comparison/; figures: report.ipynb

Every selection writes its games (PGN and `games.csv` with each game's chance to be picked and its weight), tables and a `summary.md`. The later steps use the selection named by `IE500_SELECTION` (default `stratified`).

## Preprocessing

    uv run python -m src.02_preprocessing.common.extract                   # one row per move with every column, incl. the false predictors, into data/preprocessing/extracted/
    uv run python -m src.02_preprocessing.common.leakage_check             # the kept columns are known before the move; the false predictors are not
    uv run python -m src.02_preprocessing.without_stockfish.preprocess     # removes the 13 false predictors
    uv run python -m src.02_preprocessing.without_stockfish.inspection     # missing values, near-constant columns, correlations (training games only)
    uv run python -m src.02_preprocessing.without_stockfish.reduce         # removes the columns we do not need, each with its reason; see report.ipynb
    uv run python -m src.02_preprocessing.with_stockfish.preprocess        # adds the 15 Stockfish columns (downloaded by default)
    uv run python -m src.02_preprocessing.with_stockfish.inspection
    uv run python -m src.02_preprocessing.with_stockfish.reduce

Results: `data/preprocessing/<without|with>_stockfish/final/` (38 → 30 columns without Stockfish, 53 → 37 with). Every removed column and its reason (domain knowledge or inspection) is in `removed.csv` and in the `REMOVE` list of `reduce.py`. To run Stockfish yourself instead of downloading it: `src/02_preprocessing/with_stockfish/README.md`.

Every feature is known before the move is played: the position, the clocks, the ratings, the earlier moves, the hour the game started, lobby/arena/swiss, and the opening letter from ply 12 on. Stockfish looks at the moves the player can choose from, never at the move that was played, in two passes: a deep search of the best 5 moves and a quick shallow look at every legal move for the worst 5 and the share of losing moves (38 ms per position on one core: about 15 hours for all games on a computer with 16 cores, or use the cluster, see below). A stopped engine run continues where it was.

The leakage check cuts sample games at a random move, scrambles everything Lichess wrote about that move (clock, evaluation, mark) and removes the result; the features up to the cut must stay the same. With `--engine N` it does the same for the Stockfish columns of N games.

## Training

Every method has its own folder in `src/04_data_mining/`, every run writes `data/runs/<method>/<name>/` (predictions of the held-out training games and of the test games, settings, model). Training reads the final tables of the preprocessing (downloaded if missing, see above). Feature sets: `basic` (speed, Elo, clock), `context` (every column of `data/preprocessing/without_stockfish/final/`), `engine` (every column of `data/preprocessing/with_stockfish/final/`). The columns come from the tables, so a column removed in `reduce.py` is gone from training too. Encoding: `src/03_transformation/`. `bash run_models.sh` runs every training and validation of the paper.

    uv run python -m src.04_data_mining.bucket_rate.train --source month               # baseline: the blunder probability of the move's bucket in the month
    uv run python -m src.04_data_mining.bucket_rate.train --source train               # the same from the training games only
    uv run python -m src.04_data_mining.baseline.train                                  # simpler baselines: overall rate, per speed, speed x Elo, win-probability band, after an opponent's mistake
    uv run python -m src.04_data_mining.naive_bayes.train --features context            # lower reference among the learners
    uv run python -m src.04_data_mining.logistic_regression.train --features context
    uv run python -m src.04_data_mining.svm.train --features context                    # linear SVM, probabilities by Platt scaling
    uv run python -m src.04_data_mining.decision_tree.train --features context          # symbolic: tunes the depth, writes rules.txt
    uv run python -m src.04_data_mining.random_forest.train --features context
    uv run python -m src.04_data_mining.gradient_boosting.train --features context
    uv run python -m src.04_data_mining.gradient_boosting.train --features engine
    uv run python -m src.04_data_mining.xgboost.train --features context                # --device cuda for an NVIDIA GPU
    uv run python -m src.04_data_mining.neural_network.train --features context         # early stopping on the held-out games

Hyperparameters are command-line options (`--help` of each method); give every setting its own `--name`, e.g. `--C 0.1 --name context_C0.1`, and choose between them with the validation below. Why each method is there: Chapter 4 of the final paper.

`--max-games N` trains on a random subset of the training games for a quick run. A new method goes into a new folder `src/04_data_mining/<method>/train.py` with a `fit` function (and `DEFAULTS`, `SPACE`, `GRID` for the tuning) and saves its run with `save_run` of `src/04_data_mining/common.py`. A new text column needs its values in `CATEGORIES` of `src/03_transformation/encoding/categories.py`.

### Tuning, ablation, importance

    uv run python -m src.04_data_mining.tune --method xgboost --features context        # Optuna search, then the best settings refitted as xgboost/context_tuned
    uv run python -m src.04_data_mining.tune --method gradient_boosting --sampler grid  # every combination of the paper's Table 4.2 instead
    uv run python -m src.04_data_mining.ablation --method xgboost --features context    # every feature group left out, and alone: runs ablation/<method>_<features>_<variant>
    uv run python -m src.04_data_mining.importance                                      # permutation importance by feature group of the best run (after the validation)

The tuner fits every trial on a random subset of the fitting games (`tuning.tune_moves`, about 1 million moves) and scores it on the held-out games (`tuning.objective`, log loss; `--objective pr_auc` also works); every trial goes into `tuning.csv`. The best settings are then fitted on all fitting moves, except random forest, neural network and SVM, which get a random subset (`tuning.refit_max_moves` in `config.yaml`). The ablation takes the settings of the tuned run if there is one. Methods: `naive_bayes`, `logistic_regression`, `decision_tree`, `random_forest`, `gradient_boosting`, `xgboost`, `neural_network`, `svm`.

## Validation

    uv run python -m src.05_evaluation.validate                             # scores every run in data/runs/ on the same moves, into data/validation/
    uv run python -m src.05_evaluation.validate --best gradient_boosting/context --bootstrap 200   # another run for the error analysis, fewer resamples

Choose between runs on the held-out training games (valid), report the test games. Runs are compared on the moves that all of them predicted, against the baseline `bucket_rate/month`. Besides the probability scores (log loss, Brier, ROC-AUC, PR-AUC, calibration) every run gets a cut-off for flagging a move, chosen on its own held-out games and only then used on the test games, with precision, recall, F1, MCC and balanced accuracy there, and the blunders among the 5% riskiest moves. The test scores come with 95% intervals that resample whole games. `figures/` draws the baseline and the best run of every method (ROC, precision-recall, calibration, gains and lift, scores by speed, Elo, move number, time left and position); `errors/` shows where the best run misses blunders and raises false alarms. Settings: `evaluation:` in `config.yaml`; start with `summary.md`.

## Balancing experiment

Does balancing the training games help the models? Run A is the normal pipeline above with the stratified selection in `data/`. Run B uses the balanced selection and goes to its own folder `data_balanced/`. The balanced selection keeps the test games of the stratified one, so both runs are scored on the same test games. The cleanup in `cache/` is shared; do not rerun it.

In Git Bash (in PowerShell set `$env:IE500_DATA = "data_balanced"; $env:IE500_SELECTION = "balanced"` once instead):

    export IE500_DATA=data_balanced IE500_SELECTION=balanced
    uv run python -m src.01_selection.preparation.blunder_rates.rates
    uv run python -m src.01_selection.sampling.stratified_balanced.select
    uv run python -m src.02_preprocessing.common.extract
    IE500_ENGINE_REUSE=data uv run python -m src.02_preprocessing.with_stockfish.engine   # after run A's engine: only run B's new games
    uv run python -m src.02_preprocessing.without_stockfish.preprocess
    uv run python -m src.02_preprocessing.without_stockfish.reduce            # the same columns as run A
    uv run python -m src.02_preprocessing.with_stockfish.preprocess
    uv run python -m src.02_preprocessing.with_stockfish.reduce
    uv run python -m src.04_data_mining.bucket_rate.train --source month
    uv run python -m src.04_data_mining.gradient_boosting.train --features context    # and every other run of run A, same settings
    unset IE500_DATA IE500_SELECTION

Then, back in `data/`:

    uv run python -m src.05_evaluation.validate --with balanced=data_balanced     # into data/validation/with_balanced/

The runs of run B appear as `balanced:<method>/<name>` next to those of run A, scored on the shared test games only (the held-out games differ between the two). The validation stops if the test games of the two folders differ.

## Cluster

Stockfish on every move takes about 250 core-hours. We ran it on a 16-core desktop (about 15 hours). We also prepared the same jobs for bwUniCluster 3.0 (selection, run B and Stockfish as an array of 4 nodes that share the batches through `IE500_SHARD`). The job scripts are not part of the repository, because they hold the paths of a personal workspace.

Run B shares all test games and 93% of the training games with run A, so its engine run takes over run A's files (`IE500_ENGINE_REUSE=data`) and analyses only the about 18,000 new training games. Stockfish on the cluster is version 19 built from source (the release binary needs a newer system library than the cluster has).
