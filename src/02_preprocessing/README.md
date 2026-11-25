# 02 Preprocessing

The selected games become one row per move, with every column known before the move. One table per selection,
written to `data/02_preprocessing/<selection>/`:

| script | what it does |
|---|---|
| `extract.py` | one row per move with every column we can read from a game, incl. the false predictors |
| `leakage_check.py` | cuts games at a random move and checks that no kept column changes (stratified run only) |
| `stockfish.py` | Stockfish on the position before every move: 15 columns (`sf_*`), see below |
| `preprocess.py` | removes the 13 false predictors and adds the Stockfish columns |
| `inspection.py` | missing values, near-constant columns, information about blunders, correlations (training games only) |
| `reduce.py` | removes the columns we do not need, by fixed rules; `report.ipynb` shows why |

The reduction runs twice: first on the columns without Stockfish alone, then on the Stockfish columns, which may only
remove Stockfish columns. So a model without Stockfish just leaves out the `sf_*` columns and gets exactly the table
it would have had without Stockfish. The balanced run takes the stratified run's leakage check and decisions.

## Stockfish

`data.stockfish` in `config.yaml` decides where the Stockfish columns come from.

**`download` (the default): you do not run Stockfish.** We ran it once (about 15 hours on 16 cores) for every game
of the stratified and the balanced selection and put the result on our server. `preprocess.py` downloads it by
itself into `data/02_preprocessing/stockfish/`. To download it on its own:

    uv run python -m src.common.server download 02_preprocessing/stockfish

**`compute`** (only if you change the engine settings or the selection): install Stockfish, set `STOCKFISH_PATH` in
`.env`, then

    uv run python -m src.02_preprocessing.stockfish

It writes one file per 50 games into `data/02_preprocessing/work/stockfish/<part>/` and at the end adds them to
`data/02_preprocessing/stockfish/`. It skips every game that is already analysed with the same engine and settings,
so a stopped run continues, and after a change of the selection only the new games are analysed. `preprocess.py`
stops if a move has no Stockfish columns. Search depths: `engine` in `src/settings.py`. `src/pipeline.py` runs it
before `preprocess.py` when `data.stockfish` is `compute`. On a cluster, `IE500_SHARD="i/n"` runs every n-th batch;
after the last shard, `--merge` adds the batches to the result.
