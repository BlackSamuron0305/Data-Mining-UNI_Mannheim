"""The columns we remove because they are only known after the move or after the game, or say who played."""
FALSE_PREDICTORS = {
    "mark": ("after the move", "Lichess's mark of this move (?!, ?, ??): the ?? mark is the target itself"),
    "loss": ("after the move", "how much this move lost; a blunder is defined by it"),
    "eval_after": ("after the move", "the evaluation after this move shows whether it was a blunder"),
    "clock_after": ("after the move", "the clock after this move"),
    "spent": ("after the move", "the time spent on this move"),
    "result": ("after the game", "who won; a blunder often decides the game"),
    "rating_change": ("after the game", "rating points won or lost, follows from the result"),
    "termination": ("after the game", "how the game ended (normal or on time)"),
    "game_plies": ("after the game", "the length of the game, tells how many moves are still to come"),
    "opening": ("after the opening", "the opening's name, which names moves that are not played yet"),
    "eco_full": ("after the opening", "the opening letter at every move; we keep it only from move 12 on (eco)"),
    "mover_name": ("identifier", "the player's name: says who played, not how the move is played"),
    "opp_name": ("identifier", "the opponent's name"),
}


def drop_false_predictors(df):
    """The table without the false predictors."""
    return df.drop(columns=[c for c in FALSE_PREDICTORS if c in df.columns])
