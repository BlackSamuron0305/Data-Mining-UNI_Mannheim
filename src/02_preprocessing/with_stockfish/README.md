# With Stockfish

Stockfish looks at the position before every move and tells how hard the choice is (15 columns per move).

`data.stockfish` in `config.yaml` decides where these columns come from.

**`download` (the default): you do not run Stockfish.** We ran it once (about 15 hours on 16 cores) for every game
of the stratified and the balanced selection and put the result on our server. `preprocess.py` downloads it by
itself:

    uv run python -m src.02_preprocessing.with_stockfish.preprocess

To download it on its own: `uv run python -m src.common.server download stockfish`.

**`compute`** (only if you change the engine settings or the selection): install Stockfish, set `STOCKFISH_PATH` in
`.env`, then

    uv run python -m src.02_preprocessing.with_stockfish.engine

It writes one file per 50 games into `data/02_preprocessing/<selection>/with_stockfish/stockfish/<part>/`. It
skips every game that is already analysed with the same engine and settings (in this folder, in the other run's
folder or in the downloaded results), so a stopped run continues, and after a change of the selection only the new
games are analysed. `preprocess.py` stops if a move has no Stockfish columns. Search depths: `engine` in `src/settings.py`. `src/pipeline.py` runs it before
`preprocess.py` when `data.stockfish` is `compute`.
