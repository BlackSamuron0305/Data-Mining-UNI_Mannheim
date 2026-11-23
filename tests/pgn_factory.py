"""Small Lichess games for the tests: legal moves with a clock, an evaluation and Lichess's marks on every move."""
import chess
import chess.pgn

# knights out and back: legal for any number of plies
SHUFFLE = ["Nf3", "Nf6", "Ng1", "Ng8"]
FOOLS_MATE = ["f3", "e5", "g4", "Qh4#"]


def lichess_game(plies: int | list[str] = 6, *, speed: str = "Blitz", white_elo="1500", black_elo="1600",
                 title: str | None = None, termination: str = "Normal", result: str = "1-0",
                 site: str = "abcd1234", time_control: str = "180+0", clocks: list | None = None,
                 evals: list | None = None, blunders: tuple[int, ...] = ()) -> str:
    """PGN text of one game as Lichess exports it. clocks and evals hold one entry per ply (None: left out)."""
    moves = plies if isinstance(plies, list) else (SHUFFLE * (plies // 4 + 1))[:plies]
    n = len(moves)
    if clocks is None:
        clocks = [f"0:{(180 - 2 * (t // 2)) // 60:02d}:{(180 - 2 * (t // 2)) % 60:02d}" for t in range(n)]
    if evals is None:
        evals = ["0.2"] * n
    game = chess.pgn.Game()
    h = game.headers
    h["Event"] = f"Rated {speed} game"
    h["Site"] = f"https://lichess.org/{site}"
    h["White"], h["Black"] = "white_player", "black_player"
    h["Result"] = result
    h["UTCDate"], h["UTCTime"] = "2026.09.01", "12:34:56"
    h["WhiteElo"], h["BlackElo"] = str(white_elo), str(black_elo)
    if title:
        h["WhiteTitle"] = title
    h["TimeControl"] = time_control
    h["ECO"], h["Opening"] = "A04", "Zukertort Opening"
    h["Termination"] = termination
    node = game
    board = game.board()
    for t, san in enumerate(moves):
        move = board.parse_san(san)
        parts = []
        if evals[t] is not None:
            parts.append(f"[%eval {evals[t]}]")
        if clocks[t] is not None:
            parts.append(f"[%clk {clocks[t]}]")
        nags = {chess.pgn.NAG_BLUNDER} if t in blunders else set()
        node = node.add_variation(move, comment=" ".join(parts), nags=nags)
        board.push(move)
    return str(game) + "\n\n"
