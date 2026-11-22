# 03 Transformation

Turns the columns of a move into what each model can use. Nothing is written to disk: every training run
calls this code, and every setting (the middle value of a column, the scale, the list of categories) is learned
from the training games only, so the test games never leak in.

| file | what it does | used by |
|---|---|---|
| `encoding/categories.py` | fixes the values of the text columns (speed, event type, ECO letter) | every model |
| `encoding/booleans.py` | true/false as 1/0 | every model |
| `encoding/numeric.py` | gaps filled with the middle value plus a "was missing" column, numbers scaled (or reshaped into a bell curve), one 0/1 column per category | logistic regression, SVM, Naive Bayes, neural network |
| `encoding/trees.py` | a category becomes its number, gaps stay gaps | decision tree, random forest |

XGBoost and gradient boosting take the categories and the gaps as they are.

The new columns built from a game (clock left as a share of the starting time, win chance, earlier mistakes)
are made in `src/02_preprocessing/common/extract.py`: the earlier mistakes are counted from the loss and the mark
of every move, which are removed as false predictors right after the extraction.
