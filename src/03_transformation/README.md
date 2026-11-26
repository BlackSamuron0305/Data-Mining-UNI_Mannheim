# 03 Transformation

Turns the columns of a move into what each model can use, in two parts.

**`transform.py` writes the tables the models read** (`data/03_transformation/<selection>/`): the final tables of
step 2 with the text columns as categories with a fixed list of values, true/false as 1/0, decimals as 32-bit floats
and, for the training games, the column `fold`: the held-out games that choose between runs
(`transformation.valid_frac` in `src/settings.py`). Nothing here is learned from the data, so the same tables serve
every model. A model without Stockfish leaves out the Stockfish columns (`sf_*`).

    uv run python -m src.03_transformation.transform

**What a model learns from the games stays in its own pipeline**, fitted in every training run on the training
games only, so the test games never leak in and the tuning can choose between encodings (Naive Bayes tries standard
and quantile scaling). All of it is in `encoding.py`:

| function | what it does | used by |
|---|---|---|
| `add_categories` | fixes the values of the text columns (speed, ECO letter) | `transform.py`, every model |
| `features` | true/false as 1/0 | every model |
| `fold_of` | the held-out training games | `transform.py` |
| `numeric_prep` | gaps filled with the middle value plus a "was missing" column, numbers scaled (or reshaped into a bell curve), one 0/1 column per category | logistic regression, SVM, Naive Bayes, neural network |
| `tree_features` | a category becomes its number, gaps stay gaps | decision tree, random forest |

XGBoost and gradient boosting take the categories and the gaps as they are.

The new columns built from a game (clock left as a share of the starting time, win chance, earlier mistakes)
are made in `src/02_preprocessing/moves/extract.py`: the earlier mistakes are counted from the loss and the mark of every
move, which are removed as false predictors right after the extraction.
