"""The fixed settings of the method: the month, the selection, the features, the engine, the transformation, the
tuning and the evaluation. They define the results of the paper and are not meant to be changed. What can be chosen
(where the data comes from, the number of processes) is in config.yaml."""

SETTINGS = {
    "seed": 42,
    "month": "2026-09",                          # rated standard games of September 2026
    "compare_month": "2026-08",                  # only to show that the blunder probability changes by month

    # step 1: complete cases
    "filters": {
        "speeds": ["bullet", "blitz", "rapid", "classical"],
        "terminations": ["Normal", "Time forfeit"],
    },

    # step 1: a group is speed x mean Elo, a bucket is speed x mover's Elo x time left
    "selection": {
        "n_games": 350_000,                      # training + test games
        "test_frac": 0.20,                       # share of test games in every group
        "elo_bin": 250,
        "elo_min": 500,
        "elo_max": 2000,
        "window": 30,                            # time left in 30 s steps, up to:
        "window_max": {"bullet": 120, "blitz": 300, "rapid": 900, "classical": 1800},
        "min_games": 100,                        # buckets with fewer games are marked as uncertain
        "pool_rate": 0.10,                       # share of the complete games kept to draw the selections from
        "check_draws": 20,                       # fresh draws that show how far a selection strays by chance
    },

    # step 2: what is known before the move
    "features": {
        "recent_moves": 5,                       # moves for the recent averages
        "eco_from_ply": 12,                      # the opening letter only from this ply on
        "marks": {"inaccuracy": 0.05, "mistake": 0.10, "blunder": 0.15},   # loss in win probability of the marks
    },

    # step 2: which columns the reduction removes (src/02_preprocessing/common/reduction.py)
    "reduction": {
        "apply": True,                           # False: keep every column except the false predictors
        "inspection_games": 70_000,              # training games the inspection measures on (20% of the selection)
        "redundant": 0.9,                        # two columns with |Spearman| at least this repeat each other
        "weak_share": 0.01,                      # less than 1% of the best column's information about blunders
    },

    # step 2: Stockfish on the position before every move
    "engine": {
        "top": {"limit": {"depth": 10}, "lines": 5},   # the best 5 moves
        "worst": {"limit": {"depth": 4}},              # every legal move, for the worst 5
    },

    # step 3
    "transformation": {
        "valid_frac": 0.10,                      # training games held out to choose between runs
    },

    # step 4
    "tuning": {
        "sampler": "tpe",                        # tpe or grid
        "objective": "log_loss",
        "tune_moves": 1_000_000,                 # fitting moves per trial
        "trials": {"naive_bayes": 25, "logistic_regression": 10, "decision_tree": 20, "random_forest": 16,
                   "gradient_boosting": 25, "xgboost": 30, "neural_network": 20, "svm": 20},
        "refit_max_moves": {"random_forest": 5_000_000, "neural_network": 5_000_000, "svm": 5_000_000},
        "cv_check": 5,                           # folds to compare the best two settings; 0 = off
        "ablation_max_moves": 5_000_000,
        "importance_moves": 200_000,
        "importance_repeats": 3,
    },

    # step 5
    "evaluation": {
        "top_share": 0.05,                       # scores among the 5% riskiest moves
        "bootstrap": 1000,                       # resamples of the test games for the 95% intervals
        "plot_runs": 6,
        "min_moves": 1000,                       # smaller subgroups are left out
        "move_bands": [10, 20, 30, 40],
        "clock_bands": [0.1, 0.25, 0.5, 0.75],
        "eval_bands": [0.1, 0.3, 0.7, 0.9],
    },
}
