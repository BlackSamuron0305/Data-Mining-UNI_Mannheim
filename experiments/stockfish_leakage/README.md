# Stockfish leakage

Does any feature get information about the move it predicts that is not known before that move, through Stockfish?
Stockfish enters our data in two places: our own `sf_*` columns (step 2, `engine/stockfish.py`) and Lichess's
evaluations (`eval_wp` and the history columns), which come from the same server analysis that makes the ?? label.
Lichess's analysis client (fishnet) analyses a game backwards, in chunks of 6 positions that share the engine's hash
(fishnet `src/queue.rs`, `src/stockfish.rs`), so its evaluation before a move may have seen the positions after it.

Result: no leakage that helps the models. Details and numbers: `results/`.

| test | script | question | result |
|---|---|---|---|
| E | `src/02_preprocessing/moves/leakage_check.py` | do the features see the played move? The check now also replaces the move at the cut by another legal move | no feature changed, 2,000 games, 10 with Stockfish (`results/leakage_check_swap.json`) |
| D, A | `label_coupling.py` | how the label follows from Lichess's evaluations; does `eval_wp` tell more than our `sf_wp_best`? | rule loss >= 0.15 matches 96.6% of the ?? marks; `eval_wp` a little more informative |
| B | `independent_engine.py` | XGBoost with Lichess's evaluations, with `eval_wp` from our engine, with evaluation and history from our engine | PR-AUC 0.300, 0.291, 0.272 (fixed settings, 60,000 training games, all test games) |
| hash order | `hash_order.py`, `hash_order_analysis.py` | does the backwards analysis put the played move into the evaluation before it? | 300 games at 500,000 nodes: 0.2 points lower before a blunder, no gain in prediction |
| C | `double_blunders.py` | are pairs of ?? in a row one wrong evaluation in between? | 1.3% of the pairs look like it; where our engine agrees, the rate after an opponent's ?? stays 28% |

The drop in B comes from Lichess's stronger engine (about 857,000 nodes per position against our depth 10), not from
leakage: a clean forward analysis reproduces Lichess's `eval_wp` as closely as the backwards one.

## Run

They read the stratified run's tables of step 2 and its games (`uv run python -m src.common.server download
02_preprocessing/stratified 02_preprocessing/stockfish 01_selection/stratified`) and write to
`data/experiments/stockfish_leakage/`. From the repository:

    uv run python -m src.02_preprocessing.moves.leakage_check --games 2000 --engine 10   # E, needs Stockfish
    uv run python -m experiments.stockfish_leakage.label_coupling train                 # D, A (a few minutes)
    uv run python -m experiments.stockfish_leakage.independent_engine                   # B (about 30 minutes)
    uv run python -m experiments.stockfish_leakage.hash_order --games 300 --nodes 500000 # hash order, needs Stockfish (about 2 hours on 6 cores)
    uv run python -m experiments.stockfish_leakage.hash_order_analysis
    uv run python -m experiments.stockfish_leakage.double_blunders train                # C

`results/leakage_check_server.json` is the check as it was on our server before (no Stockfish games).

Limits: 300 games and 500,000 nodes instead of Lichess's 857,000; Stockfish 19 instead of Lichess's 17.1 and 18;
the engine part of E covers 10 games; only the stratified run was tested.
