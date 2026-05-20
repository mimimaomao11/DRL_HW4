"""Tool 3: Result Analyzer — loads experiment_db.json and produces ranked comparison and insights."""

import json
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "memory" / "experiment_db.json"


def analyze_results(experiment_ids: list[str], metrics: list[str] | None = None) -> dict:
    """Analyze and compare one or more RL experiments from the persistent database.

    Args:
        experiment_ids: List of experiment ID strings, or ["all"] to analyze every run.
        metrics: Optional list of metric names to surface (default: mean_reward, std_reward).

    Returns:
        dict with comparison_table (sorted best→worst), best_experiment, insights list.
    """
    if not DB_PATH.exists():
        return {
            "error": "No experiments found. Run at least one experiment first.",
            "comparison_table": [],
        }

    with open(DB_PATH, encoding="utf-8") as f:
        db = json.load(f)

    if not db:
        return {"error": "Experiment database is empty.", "comparison_table": []}

    # Resolve IDs
    if "all" in experiment_ids:
        selected = list(db.values())
    else:
        selected = [db[eid] for eid in experiment_ids if eid in db]
        missing = [eid for eid in experiment_ids if eid not in db]
        if missing:
            return {
                "error": f"Unknown experiment IDs: {missing}",
                "available_ids": list(db.keys()),
            }

    completed = [e for e in selected if e.get("status") == "completed"]
    if not completed:
        return {
            "error": "All selected experiments have status 'failed'.",
            "failed_experiments": selected,
        }

    metrics = metrics or ["mean_reward", "std_reward"]

    table = []
    for exp in completed:
        row = {"experiment_id": exp["experiment_id"], "algorithm": exp["algorithm"],
               "environment": exp["environment"], "n_steps": exp.get("n_steps")}
        for m in metrics:
            row[m] = exp.get(m, "N/A")
        table.append(row)

    table.sort(key=lambda x: x.get("mean_reward", float("-inf")), reverse=True)

    return {
        "total_experiments": len(table),
        "best_experiment": table[0] if table else None,
        "comparison_table": table,
        "insights": _generate_insights(table),
    }


def _generate_insights(table: list[dict]) -> list[str]:
    if not table:
        return []
    insights = []

    best = table[0]
    insights.append(
        f"Best run: {best['algorithm']} on {best['environment']} "
        f"— mean reward {best.get('mean_reward', '?')} ± {best.get('std_reward', '?')}"
    )

    if len(table) > 1:
        worst = table[-1]
        insights.append(
            f"Lowest run: {worst['algorithm']} on {worst['environment']} "
            f"— mean reward {worst.get('mean_reward', '?')}"
        )

    # Per-environment best algorithm
    envs = {e["environment"] for e in table}
    for env in sorted(envs):
        env_rows = [e for e in table if e["environment"] == env]
        if len(env_rows) > 1:
            champion = max(env_rows, key=lambda x: x.get("mean_reward", float("-inf")))
            insights.append(
                f"On {env}: best algorithm is {champion['algorithm']} "
                f"({champion.get('mean_reward', '?')})"
            )

    return insights
