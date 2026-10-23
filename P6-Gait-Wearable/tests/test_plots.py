from pathlib import Path

from plots import make_figures
from test_train import write_session
from train import LABELS


def test_make_figures_writes_four_pngs(tmp_path):
    data, out = tmp_path / "data", tmp_path / "fig"
    data.mkdir()
    for i, kind in enumerate(LABELS):
        write_session(data / ("20261023-1200%02d-%s.csv" % (i, kind.lower())), [kind], seconds=6.0)
    written = make_figures(sorted(data.glob("*.csv")), out)
    assert sorted(p.name for p in written) == ["confusion.png", "features.png", "tree_model1.png", "tree_model2.png"]
    assert all(p.stat().st_size > 5000 for p in written)
