"""Test C: are pairs of ?? in a row (the opponent's move, then the mover's) partly an artefact of one wrong
evaluation in Lichess's analysis?

If Lichess's engine overrates the position between the two moves (a spike), the opponent's move looks like a big
loss and the mover's next move looks like a big loss too, although the position hardly changed: two ?? marks from
one wrong number. That number feeds a feature (opp_last_loss) and the label at the same time.

A spike shows as: Lichess's eval of the position in between (eval_wp at ply t) far above our own engine's view of
the same position (sf_wp_best at ply t, depth 10, independent), and the evaluation coming back to where it was.

Usage (from the repository): uv run python -m experiments.stockfish_leakage.double_blunders [train|test]
"""
import sys

import numpy as np
import pandas as pd

part = sys.argv[1] if len(sys.argv) > 1 else "train"
df = pd.read_parquet(f"data/02_preprocessing/stratified/{part}.parquet",
                     columns=["game_id", "ply", "blunder", "eval_wp", "opp_last_loss", "opp_blunders"])
sf = pd.read_parquet(f"data/02_preprocessing/stockfish/{part}.parquet", columns=["game_id", "ply", "sf_wp_best"])
df = df.merge(sf, on=["game_id", "ply"], validate="one_to_one").sort_values(["game_id", "ply"]).reset_index(drop=True)

same_prev = df["game_id"].eq(df["game_id"].shift(1))
df["opp_blunder_before"] = same_prev & df["blunder"].shift(1, fill_value=False)   # the opponent's last move was ??
df["r"] = df["eval_wp"] - df["sf_wp_best"]                                        # Lichess minus ours, at ply t
# the mover's win probability two plies earlier (before the opponent's move), seen from the mover
df["wp_two_before"] = np.where(same_prev & df["game_id"].eq(df["game_id"].shift(2)), 1 - df["eval_wp"].shift(1), np.nan)
same_next = df["game_id"].eq(df["game_id"].shift(-1))
df["wp_after"] = np.where(same_next, 1 - df["eval_wp"].shift(-1), np.nan)          # mover, after the move

after_opp = df[df["opp_blunder_before"]]
print(f"{part}: {len(df):,} moves, blunder rate {df['blunder'].mean():.4f}")
print(f"moves right after an opponent's ??: {len(after_opp):,}, of them ??: {after_opp['blunder'].mean():.4f}")

pairs = after_opp[after_opp["blunder"]]
print(f"\npairs ?? then ??: {len(pairs):,} ({len(pairs) / df['blunder'].sum():.1%} of all ??)")

big = 0.15
print(f"\nshare of moves where Lichess's eval before the move is more than {big} above ours (r > {big}):")
for name, g in [("all moves", df), ("all ??", df[df["blunder"]]), ("after an opponent's ??, no ??", after_opp[~after_opp["blunder"]]),
                ("pairs ?? then ??", pairs)]:
    print(f"   {name:32s} n {len(g):>10,}  r > {big}: {(g['r'] > big).mean():.3f}   mean r {g['r'].mean():+.4f}")

# spike: Lichess high, ours not, and the evaluation comes back to (near) where it was two plies before
spike = pairs[(pairs["r"] > big) & ((pairs["wp_after"] - pairs["wp_two_before"]).abs() < 0.10)]
print(f"\npairs that look like one wrong number (r > {big} and the evaluation back within 0.10): {len(spike):,} "
      f"= {len(spike) / len(pairs):.1%} of the pairs, {len(spike) / df['blunder'].sum():.2%} of all ??")

# what our own engine says about the move after the opponent's ??: does the rate stay high with our evaluation?
agree = after_opp[after_opp["r"].abs() <= 0.05]
print(f"\nafter an opponent's ??, where our engine agrees with Lichess (|r| <= 0.05): n {len(agree):,}, "
      f"?? rate {agree['blunder'].mean():.4f} (all: {after_opp['blunder'].mean():.4f})")
for lo, hi in [(-1, -0.15), (-0.15, -0.05), (-0.05, 0.05), (0.05, 0.15), (0.15, 1)]:
    g = after_opp[(after_opp["r"] > lo) & (after_opp["r"] <= hi)]
    print(f"   r in ({lo:+.2f}, {hi:+.2f}]: n {len(g):>8,}, ?? rate {g['blunder'].mean():.4f}")
