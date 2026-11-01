"""Static results site: baked JSON, complete runs only, no optima, live episode hidden."""
import json
from pathlib import Path

from sim2gym.site import ROUTES, build_site

REPO = Path(__file__).resolve().parents[1]


def test_build_site(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for name in ("oracle", "6-probe"):
        (results / f"{name}.jsonl").write_text((REPO / "results" / f"{name}.jsonl").read_text())
    rows = (REPO / "results" / "random.jsonl").read_text().splitlines()
    (results / "partial.jsonl").write_text("\n".join(rows[:5]) + "\n")  # incomplete run: left out
    info = build_site(tmp_path / "site", REPO / "tasks.jsonl", results, "test")
    assert info["runs"] == ["6-probe", "oracle"]
    site = tmp_path / "site"
    for path in ROUTES.values():
        assert (site / path).exists()
    assert "optimum" not in (site / ROUTES["/tasks/default"]).read_text()
    lb = json.loads((site / ROUTES["/leaderboard"]).read_text())
    assert {r["model"] for r in lb["rows"]} >= {"oracle", "6-probe"} and "random" not in {r["model"] for r in lb["rows"]}
    page = (site / "index.html").read_text()
    assert page.count("window.fetch = ") == 1 and page.index("window.fetch = ") < page.index("async function init")
    assert "#tab-live" in page
