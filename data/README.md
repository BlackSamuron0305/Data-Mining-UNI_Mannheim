# data

Everything here is built by the scripts in `src/preprocessing/` and is not tracked by git, apart from this file.

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

The working files of the selection (the full count of September and the pool of candidate games, about 650 MB) are in `cache/census/`, which is not tracked by git either.
