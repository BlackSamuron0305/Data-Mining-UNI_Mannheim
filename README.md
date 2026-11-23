# Data-Mining-UNI_Mannheim

IE500 Data Mining, University of Mannheim. Team 5: Predicting Blunders in Online Chess.

## Layout

- `IE500_DM_StudentProject.pdf`: the course's project description
- `submission/`: proposal, final paper and presentation (LaTeX)
- `src/`: the code in the five steps of the data mining process (Fayyad et al., 1996); `src/settings.py` holds the fixed settings of the method, `config.yaml` what can be chosen (where the data comes from, the number of processes), and `src/pipeline.py` runs steps 1 to 3 as `config.yaml` says
  - `src/01_selection/`: getting the data and choosing the games
    - `download/`: the raw files of September and August, from our server or from Lichess (`storage.py`)
    - `preparation/cleanup/`: one pass over the month that drops unwanted games, counts moves and blunders per bucket and keeps a pool of games
    - `preparation/blunder_rates/`: the blunder probability per bucket of the month, with `plots.ipynb`; `months.py` compares September with August, the only use of August: the blunder probability changes from month to month (a monthly bias)
    - `sampling/`: the `random`, `stratified` and `stratified_balanced` selection, each in its own folder, the code they share in `common/`, and `comparison/`: the three selections against the month, with `report.ipynb` (incl. the weights)
  - `src/02_preprocessing/`: the selected games turned into one row per move, the false predictors removed and the columns reduced
    - `common/`: the extraction (every column, incl. the false predictors), the list of false predictors, the leakage check, the inspection
    - `without_stockfish/`: preprocess, inspect (`report.ipynb`: correlation heatmap and decisions), reduce
    - `with_stockfish/`: the same with the 15 Stockfish columns; Stockfish comes from the team's server by default (its `README.md`)
  - `src/03_transformation/`: `transform.py` writes the tables the models read (fixed categories, 0/1, the held-out games); `encoding/` holds what each model learns from the training games: gaps filled, numbers scaled, categories as 0/1 columns or as numbers for the trees (its `README.md`)
  - `src/04_data_mining/<method>/`: one folder per training method; `src/04_data_mining/common.py` holds what they share
  - `src/05_evaluation/`: scores every training run on the same moves
  - `src/common/`: code used by several steps
    - `pgn.py`: reads a Lichess game (moves, clocks, evaluations, blunder marks); steps 1 and 2
    - `server.py`: lists, downloads and uploads our prepared data on the team's server; steps 1 and 2, and on its own
    - `buckets.py`: the groups of games and buckets of moves; steps 1 to 5
    - `metrics.py`: the scores of the predictions; steps 4 and 5
    - `plot_style.py`: colours and sizes of the figures; the notebooks of steps 1 and 2
- `data/`: everything the code builds, one folder per step like `src/` (`01_selection` to `05_evaluation`), not tracked by git (see `data/README.md`)
- `tests/`: tests of the code on small made-up games, no data needed: `uv run pytest` (about 15 seconds)

## Setup

We use [uv](https://docs.astral.sh/uv/) as the package manager, not pip. In the repository run

    uv sync

once. It installs Python 3.14 and the exact package versions of `uv.lock` into `.venv`. Run every script with `uv run`. Add a package with `uv add <name>`. `pyproject.toml`, `uv.lock` and `.python-version` are tracked by git; `.venv` is not.

Copy `.env.example` to `.env`. It holds only the keys of our server (needed to download) and, if you have them, the paths of the raw month and of Stockfish on your computer. What to run is set in `config.yaml`; the settings of the method itself (selection, features, engine, transformation, tuning, evaluation) are fixed in `src/settings.py`.

## Where the data comes from

`data:` in `config.yaml` chooses where the pipeline starts. The chosen state is downloaded from our server; every step after it is computed:

| `start_with` | downloaded | computed |
|---|---|---|
| `raw` | nothing: the raw month comes from Lichess (`raw_from: lichess`, about 30 GB) or from our copy (`raw_from: server`) | steps 1, 2 and 3 |
| `selected` | the selected games, after step 1 (selection) | steps 2 and 3 |
| `preprocessed` | the move tables, after step 2 (preprocessing) | step 3 |
| `transformed` | the tables ready for the models, after step 3 (transformation) | nothing |

`also_download` downloads earlier states as well, to look at them, e.g. `also_download: [selected, preprocessed]`. `stockfish` decides where the Stockfish columns of step 2 come from: `download` (our results) or `compute` (about 15 hours on 16 cores; needs `STOCKFISH_PATH` in `.env`, see `src/02_preprocessing/with_stockfish/README.md`). Then

    uv run python -m src.pipeline --dry-run   # what would be downloaded and computed
    uv run python -m src.pipeline             # fills data/ up to step 3
    bash run_models.sh                        # steps 4 and 5: training and evaluation

A step is skipped if its result is there and newer than everything it reads, so a stopped run continues and a change (e.g. of the filters in `src/settings.py`) recomputes only what depends on it; the cleanup also checks that it was made with the current filters. On a 16-core desktop the cleanup of the month takes about 20 minutes and the selections about a minute each; the extraction of the moves takes a few minutes per run.

Every script also runs on its own (the sections below). What it needs and does not find, it downloads if `config.yaml` takes that state from our server; otherwise it stops and names the step to run first. Downloads are checked against their sha256 and go to the same folders as the computed files, so the later steps cannot tell the difference.

`uv run python -m src.common.server list` shows what is on the server; `download` fetches the states of `config.yaml` (or the ones named). `upload selected|preprocessed|transformed|stockfish` puts a state computed on this computer there and needs a key that may write; `stockfish` is the Stockfish columns of both runs.

## Data selection

    uv run python -m src.01_selection.download.september                 # the raw file of September 2026 into data/01_selection/raw/<month>/ (about 30 GB, checked against Lichess's sha256)
    uv run python -m src.01_selection.download.august                    # the same for August 2026
    IE500_MONTH=2026-08 uv run python -m src.01_selection.preparation.cleanup.cleanup --no-pool   # August: counts only, for its blunder rates
    IE500_MONTH=2026-08 uv run python -m src.01_selection.preparation.blunder_rates.rates
    uv run python -m src.01_selection.preparation.blunder_rates.months   # after the rates of both months: September against August
    uv run python -m src.01_selection.preparation.cleanup.cleanup        # one pass over the month into data/01_selection/preparation/cleanup/<month>/ (about 20 minutes)
    uv run python -m src.01_selection.preparation.blunder_rates.rates    # blunder probability per bucket into data/01_selection/preparation/blunder_rates/<month>/; figures: plots.ipynb
    uv run python -m src.01_selection.sampling.random.select             # 350,000 games at random into data/01_selection/sampling/random/
    uv run python -m src.01_selection.sampling.stratified.select         # stratified by speed x Elo, test games balanced, into data/01_selection/sampling/stratified/
    uv run python -m src.01_selection.sampling.stratified_balanced.select  # the same, training games balanced too, into data/01_selection/sampling/stratified_balanced/
    uv run python -m src.01_selection.sampling.comparison.compare  # the three against the month into data/01_selection/sampling/comparison/; figures: report.ipynb

Every selection writes its games (PGN and `games.csv` with each game's chance to be picked and its weight), tables and a `summary.md`. The later steps use the selection named by `IE500_SELECTION` (default `stratified`).

## Preprocessing

    uv run python -m src.02_preprocessing.common.extract                   # one row per move with every column, incl. the false predictors, into data/02_preprocessing/<selection>/extracted/
    uv run python -m src.02_preprocessing.common.leakage_check             # the kept columns are known before the move; the false predictors are not
    uv run python -m src.02_preprocessing.without_stockfish.preprocess     # removes the 13 false predictors
    uv run python -m src.02_preprocessing.without_stockfish.inspection     # missing values, near-constant columns, correlations (training games only)
    uv run python -m src.02_preprocessing.without_stockfish.reduce         # removes the columns we do not need, by fixed rules; see report.ipynb
    uv run python -m src.02_preprocessing.with_stockfish.preprocess        # adds the 15 Stockfish columns (data.stockfish: downloaded or computed)
    uv run python -m src.02_preprocessing.with_stockfish.inspection
    uv run python -m src.02_preprocessing.with_stockfish.reduce

Results: `data/02_preprocessing/<selection>/<without|with>_stockfish/final/` (38 features after the false predictors, 53 with Stockfish). The reduction follows fixed rules on the inspection of 70,000 training games (`src/02_preprocessing/common/reduction.py`, settings `reduction` in `src/settings.py`): *domain*: a column that follows from others or that a kept column covers (a written list in each `reduce.py`); *redundant*: of two columns with |Spearman| ≥ 0.9 the one that tells less about blunders (mutual information); *weak*: less than 1% of what the best column tells. A short list of columns stays whatever the numbers say (`KEEP`, with reasons). With Stockfish only Stockfish columns can go, so that table is the one without plus Stockfish columns; the balanced run takes the stratified run's decisions. Every removed column, its rule and the measured numbers are in `removed.csv`, the kept exceptions in `kept.csv`. `reduction.apply: False` keeps every column but the false predictors. To run Stockfish yourself instead of downloading it: `src/02_preprocessing/with_stockfish/README.md`.

## Transformation

    uv run python -m src.03_transformation.transform                     # the tables the models read, into data/03_transformation/<selection>/

Fixed categories, true/false as 1/0, 32-bit floats and the column `fold` (the held-out training games, `transformation.valid_frac`). What a model learns from the games (gaps filled, scaling, one 0/1 column per category) is fitted inside each training run, on the training games only (`src/03_transformation/README.md`).

Every feature is known before the move is played: the position, the clocks, the ratings, the earlier moves, the hour the game started, lobby/arena/swiss, and the opening letter from ply 12 on. Stockfish looks at the moves the player can choose from, never at the move that was played, in two passes: a deep search of the best 5 moves and a quick shallow look at every legal move for the worst 5 and the share of losing moves (38 ms per position on one core: about 15 hours for all games on a computer with 16 cores, or use the cluster, see below). A stopped engine run continues where it was.

The leakage check cuts sample games at a random move, scrambles everything Lichess wrote about that move (clock, evaluation, mark) and removes the result; the features up to the cut must stay the same. With `--engine N` it does the same for the Stockfish columns of N games.

## Training

Every method has its own folder in `src/04_data_mining/`, every run writes `data/04_data_mining/<selection>/<method>/<name>/` (predictions of the held-out training games and of the test games, settings, model). Training reads the tables of step 3 (downloaded if `config.yaml` says so, see above). Feature sets: `basic` (speed, Elo, clock), `context` (every column of `data/03_transformation/<selection>/without_stockfish/`), `engine` (every column of `data/03_transformation/<selection>/with_stockfish/`). The columns come from the tables, so a column removed in `reduce.py` is gone from training too. `bash run_models.sh` runs every training and validation of the paper.

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

The tuner fits every trial on a random subset of the fitting games (`tuning.tune_moves`, about 1 million moves) and scores it on the held-out games (`tuning.objective`, log loss; `--objective pr_auc` also works); every trial goes into `tuning.csv`. The best settings are then fitted on all fitting moves, except random forest, neural network and SVM, which get a random subset (`tuning.refit_max_moves` in `src/settings.py`). The ablation takes the settings of the tuned run if there is one. Methods: `naive_bayes`, `logistic_regression`, `decision_tree`, `random_forest`, `gradient_boosting`, `xgboost`, `neural_network`, `svm`.

## Validation

    uv run python -m src.05_evaluation.validate                             # scores every run in data/04_data_mining/<selection>/ on the same moves, into data/05_evaluation/<selection>/
    uv run python -m src.05_evaluation.validate --best gradient_boosting/context --bootstrap 200   # another run for the error analysis, fewer resamples

Choose between runs on the held-out training games (valid), report the test games. Runs are compared on the moves that all of them predicted, against the baseline `bucket_rate/month`. Besides the probability scores (log loss, Brier, ROC-AUC, PR-AUC, calibration) every run gets a cut-off for flagging a move, chosen on its own held-out games and only then used on the test games, with precision, recall, F1, MCC and balanced accuracy there, and the blunders among the 5% riskiest moves. The test scores come with 95% intervals that resample whole games. `figures/` draws the baseline and the best run of every method (ROC, precision-recall, calibration, gains and lift, scores by speed, Elo, move number, time left and position); `errors/` shows where the best run misses blunders and raises false alarms. Settings: `evaluation` in `src/settings.py`; start with `summary.md`.

## Balancing experiment

Does balancing the training games help the models? The stratified run is the normal pipeline above with the stratified selection. The balanced run uses the balanced selection; its tables and runs go next to the stratified run's, in `data/02_preprocessing/stratified_balanced/`, `data/03_transformation/stratified_balanced/` and `data/04_data_mining/stratified_balanced/`. `src/pipeline.py` makes the tables of both runs. Both selections draw the same 350,000 games stratified and split them 80/20 in every group; the test part is then balanced to the month's blunder rate in every bucket, the same test games in both runs (as the professor advised), and the balanced run also balances its training part. The two parts take their replacement games from separate spare games, so they never share a game, and both runs are scored on the same balanced test games. The cleanup is shared; do not rerun it.

In Git Bash (in PowerShell set `$env:IE500_SELECTION = "stratified_balanced"` once instead):

    export IE500_SELECTION=stratified_balanced
    uv run python -m src.01_selection.sampling.stratified_balanced.select
    uv run python -m src.02_preprocessing.common.extract
    uv run python -m src.02_preprocessing.with_stockfish.engine           # after the stratified run's engine: only the balanced run's new games
    uv run python -m src.02_preprocessing.without_stockfish.preprocess
    uv run python -m src.02_preprocessing.without_stockfish.reduce            # the same columns as the stratified run
    uv run python -m src.02_preprocessing.with_stockfish.preprocess
    uv run python -m src.02_preprocessing.with_stockfish.reduce
    uv run python -m src.03_transformation.transform
    uv run python -m src.04_data_mining.bucket_rate.train --source month
    uv run python -m src.04_data_mining.tune --method gradient_boosting --features context --params-from stratified --cv 0   # and every other run of the stratified run, same settings
    unset IE500_SELECTION

Then:

    uv run python -m src.05_evaluation.validate --with stratified_balanced     # into data/05_evaluation/stratified/with_stratified_balanced/

The models of the balanced run appear as `balanced:<method>/<name>` next to those of the stratified run, scored on the shared test games only (the held-out games differ between the two). The validation stops if the test games of the two selections differ.

## Cluster

Stockfish on every move takes about 250 core-hours. We ran it on a 16-core desktop (about 15 hours). We also prepared the same jobs for bwUniCluster 3.0 (selection, the balanced run and Stockfish as an array of 4 nodes that share the batches through `IE500_SHARD`). The job scripts are not part of the repository, because they hold the paths of a personal workspace.

The balanced run shares all test games and 93% of the training games with the stratified run, so its engine run skips the games the stratified run already analysed and analyses only the about 18,000 new training games. Stockfish on the cluster is version 19 built from source (the release binary needs a newer system library than the cluster has).
