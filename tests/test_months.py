"""Step 1, September against August: the blunder probability of each month."""
import importlib

import pandas as pd

months = importlib.import_module("src.01_selection.months")


def fake_rates(folder, moves, blunders):
    folder.mkdir(parents=True)
    pd.DataFrame({"speed": ["bullet", "bullet", "blitz", "rapid", "classical"], "moves": moves,
                  "blunders": blunders}).to_csv(folder / "rates.csv", index=False)


def test_months_compares_the_base_rates(tmp_path, monkeypatch):
    fake_rates(tmp_path / "september", [500, 500, 1000, 1000, 1000], [30, 30, 50, 60, 40])
    fake_rates(tmp_path / "august", [1000, 1000, 1000, 1000, 1000], [65, 65, 55, 60, 50])
    monkeypatch.setattr(months, "SELECTION_ROOT", tmp_path)
    monkeypatch.setattr(months, "OUT_DIR", tmp_path / "months")
    months.run()
    table = pd.read_csv(tmp_path / "months" / "months.csv").set_index(["month", "speed"])
    assert table.loc[("2026-09", "all"), "rate"] == 210 / 4000
    assert table.loc[("2026-08", "bullet"), "rate"] == 130 / 2000
    assert (table["rate_lo"] <= table["rate"]).all() and (table["rate"] <= table["rate_hi"]).all()
    summary = (tmp_path / "months" / "months.md").read_text()
    assert "September 2026" in summary and "August 2026" in summary
    assert "| all | 5.25 |" in summary and "| +0.65 |" in summary             # August: 295 / 5000 = 5.90%
