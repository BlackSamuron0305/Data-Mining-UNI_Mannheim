"""For the models that need numbers on one scale: logistic regression, SVM, Naive Bayes, neural network.

Fill the gaps, mark where a value was missing, scale the numbers, one 0/1 column per category.
Everything is learned in fit(), so only from the training games.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from src.config import SEED
from .categories import CATEGORICAL


def scaler(name="standard"):
    """standard: mean 0 and spread 1; quantile: every column reshaped into a bell curve."""
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=SEED)
    return StandardScaler()


def numeric_prep(cols, scale=None):
    """Median fill plus missing indicators, then scaling; categorical features one-hot encoded."""
    if scale is None:
        scale = scaler()
    cat = [c for c in cols if c in CATEGORICAL]
    num = [c for c in cols if c not in CATEGORICAL]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), scale), num),
        ("missing", MissingIndicator(features="missing-only", error_on_new=False), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
