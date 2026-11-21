"""The chance of every selected game to be picked, and its weight (1 / chance)."""
import pandas as pd


def same_chance(games: pd.DataFrame, month_games: int) -> pd.DataFrame:
    """Every game of the month had the same chance."""
    games["prob"] = len(games) / month_games
    for part in ("train", "test"):
        on = games["split"] == part
        games.loc[on, "part_prob"] = on.sum() / month_games
    return add_weights(games)


def chance_by_group(games: pd.DataFrame, group_games: pd.Series) -> pd.DataFrame:
    """The chance of a game is the selected games of its group / the games of its group in the month."""
    selected = games.groupby("group").size()
    games["prob"] = games["group"].map(selected / group_games)
    for part in ("train", "test"):
        on = games["split"] == part
        selected = games[on].groupby("group").size()
        games.loc[on, "part_prob"] = games.loc[on, "group"].map(selected / group_games)
    return add_weights(games)


def add_weights(games: pd.DataFrame) -> pd.DataFrame:
    games["weight"] = 1 / games["prob"]
    games["part_weight"] = 1 / games["part_prob"]
    return games
