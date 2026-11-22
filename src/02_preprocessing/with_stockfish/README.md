# With Stockfish

Stockfish looks at the position before every move and tells how hard the choice is (15 columns per move).

**Default: you do not run Stockfish.** We ran it once (about 15 hours on 16 cores) for every game of the
stratified and the balanced selection and put the result on the team's server. With the keys in `.env`,
`preprocess.py` downloads it by itself:

    uv run python -m src.02_preprocessing.with_stockfish.preprocess

To download it on its own: `uv run python -m src.common.server download stockfish`.

**From scratch** (only if you change the engine settings or the selection): install Stockfish, set
`STOCKFISH_PATH` in `.env`, then

    uv run python -m src.02_preprocessing.with_stockfish.engine

It writes one file per 50 games into `data/preprocessing/<selection>/with_stockfish/stockfish/<part>/` and continues where it
stopped. Settings: `engine:` in `config.yaml`.
