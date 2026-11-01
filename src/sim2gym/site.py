"""Static results site: the demo page with Insights and Leaderboard baked in as JSON, no server.

    sim2gym site -o site        # then deploy site/ to any static host (e.g. `vercel deploy site --prod`)

Only complete runs on the default task set are included (a run still in progress would show partial n), and
only public task fields are written (no optima). Live episodes need the Python server, so the page hides them.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from . import tasks as tasks_mod
from .runner import task_set_hash

STATIC = Path(__file__).resolve().parent / "static"
ROUTES = {"/leaderboard": "data/leaderboard.json", "/insights": "data/insights.json",
          "/tasks/default": "data/tasks.json", "/models": "data/models.json"}

SNAPSHOT_JS = """<script>
/* Static snapshot: serve the API's GET routes from baked JSON files; live episodes need `sim2gym serve`. */
(function () {
  const ROUTES = %(routes)s, real = window.fetch.bind(window);
  window.fetch = (url, opts) => real(ROUTES[url] || url, opts);
  const css = document.createElement("style");
  css.textContent = "#tab-live, .sec:has(#model), .sec:has(#tasks) { display: none !important; }";
  document.head.appendChild(css);
  const views = document.querySelector(".views");
  if (views) views.insertAdjacentHTML("afterend",
    '<div class="side-note" style="margin-top:10px">Snapshot of saved results (%(stamp)s). Live episodes run on the ' +
    'local server: <code>sim2gym serve</code>.</div>');
})();
</script>
"""


def complete_runs(results_dir: Path, tasks: list[dict], dest: Path) -> list[str]:
    """Copy result files that cover every task of the set (by id and hash) into ``dest``."""
    ids, h = {t["id"] for t in tasks}, task_set_hash(tasks)
    kept = []
    for f in sorted(results_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        if {r["task_id"] for r in rows if r.get("task_set_hash") == h} >= ids:
            shutil.copy(f, dest / f.name)
            kept.append(f.stem)
    return kept


def build_site(out: Path, tasks_path: Path, results_dir: Path, stamp: str) -> dict:
    from fastapi.testclient import TestClient

    from .api import create_app
    tasks = tasks_mod.load_tasks(tasks_path)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="sim2gym-site-") as tmp:
        kept = complete_runs(Path(results_dir), tasks, Path(tmp))
        client = TestClient(create_app(tasks_path=tasks_path, results_dir=tmp))
        for route in ("/leaderboard", "/insights", "/tasks/default"):
            resp = client.get(route)
            resp.raise_for_status()
            (out / ROUTES[route]).write_text(json.dumps(resp.json()))
    (out / ROUTES["/models"]).write_text(json.dumps(
        {"models": [], "default": "", "allow_paid": False, "paid_guard": {"max_tasks": 0, "max_turns": 0}}))
    page = (STATIC / "index.html").read_text()
    inject = SNAPSHOT_JS % {"routes": json.dumps(ROUTES), "stamp": stamp}
    marker = "<script>"
    i = page.rindex(marker)  # the page's main script, at the end of <body>
    (out / "index.html").write_text(page[:i] + inject + page[i:])
    return {"out": str(out), "runs": kept, "task_set_hash": task_set_hash(tasks)}
