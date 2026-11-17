# Data-Mining-UNI_Mannheim

IE500 Data Mining, University of Mannheim. Team 5: Predicting Blunders in Online Chess.

## Layout

- `IE500_DM_StudentProject.pdf`: the course's project description
- `submission/`: proposal, final paper and presentation (LaTeX)
- `src/`: the code, with its settings in `config.yaml`; the data selection is in `src/preprocessing/selection/`
- `data/`: the selected games, built by the code and not tracked by git (see `data/README.md`)
- `cache/`: working files of the code, not tracked by git

## Setup

We use [uv](https://docs.astral.sh/uv/) as the package manager, not pip. In the repository run

    uv sync

once. It installs Python 3.14 and the exact package versions of `uv.lock` into `.venv`. Run every script with `uv run`. Add a package with `uv add <name>`. None of the uv files (`.venv`, `pyproject.toml`, `uv.lock`, `.python-version`) are tracked by git.

Copy `.env.example` to `.env` and fill in the keys of the Lichess storage.

## Data selection

    uv run python -m src.preprocessing.selection.census                  # counts every game of September 2026 once (about 20 minutes)
    uv run python -m src.preprocessing.selection.stratified_selection    # draws the 350,000 games into data/stratified_selection/ (about 6 minutes)
    uv run python -m src.preprocessing.selection.probabilities           # adds every game's chance to be selected and its weight
