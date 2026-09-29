"""
eval/report.py -- Section 5.3 / 6.2
"""

import json
import os
from datetime import datetime, timezone

from .retrieval_metrics import aggregate_retrieval_metrics
from .telemetry import summarize_telemetry_across_questions
from .judge import overall_score

ALL_CATEGORIES = [
    "basic", "semantic", "intra_document", "project_related", "constrained",
    "conflicting_info", "completeness", "miscellaneous", "high_level", "info_not_found",
]


def _category_specific_fields(question_type: str, rows: list[dict]) -> dict:
    extra: dict = {}
    if question_type == "constrained":
        triggered = [r["routing"] for r in rows if r["routing"].get("triggered")]
        extra["fallback_rate"] = (
            sum(1 for t in triggered if t.get("fell_back_to_unfiltered")) / len(triggered)
            if triggered else None
        )
    if question_type == "completeness":
        vals = [r["routing"]["sweep_iterations"] for r in rows if "sweep_iterations" in r["routing"]]
        extra["sweep_iterations_mean"] = sum(vals) / len(vals) if vals else None
    if question_type == "info_not_found":
        vals = [r["routing"]["false_info_not_found"] for r in rows if "false_info_not_found" in r["routing"]]
        extra["false_info_not_found_rate"] = sum(1 for v in vals if v) / len(vals) if vals else None
    return extra


def aggregate_report(rows: list[dict]) -> dict:
    by_category: dict[str, list[dict]] = {}
    for r in rows:
        by_category.setdefault(r["question_type"], []).append(r)

    category_report = {}
    for category in ALL_CATEGORIES:
        cat_rows = by_category.get(category, [])
        if not cat_rows:
            continue

        verdicts = [r["judge_verdict"] for r in cat_rows]
        retrieval_metrics = aggregate_retrieval_metrics([r["retrieval"] for r in cat_rows])
        telemetry_summary = summarize_telemetry_across_questions([r["telemetry"] for r in cat_rows])
        correct = [v.correctness for v in verdicts if v is not None]
        completeness = [v.completeness_pct for v in verdicts if v is not None]

        category_report[category] = {
            "n": len(cat_rows),
            "n_errors": sum(1 for r in cat_rows if r.get("error")),
            "document_recall_mean": retrieval_metrics["mean_raw_recall"],
            "cited_recall_mean": retrieval_metrics["mean_cited_recall"],
            "invalid_extra_docs_mean": retrieval_metrics["mean_raw_invalid_extra"],
            "correctness_rate": sum(correct) / len(correct) if correct else None,
            "completeness_mean_pct": sum(completeness) / len(completeness) if completeness else None,
            "overall_score": overall_score(verdicts) if verdicts else None,
            **telemetry_summary,
            **_category_specific_fields(category, cat_rows),
        }

    all_verdicts = [r["judge_verdict"] for r in rows]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_total": len(rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "overall_score": overall_score(all_verdicts) if all_verdicts else None,
        "per_category": category_report,
    }


def write_json_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # judge_verdict/final_state aren't part of `report` itself (only derived
    # numbers are) so this is safe to dump directly without a custom encoder
    # beyond `default=str` for any stray non-JSON-native value.
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def write_markdown_report(report: dict, path: str) -> None:
    def fmt(v, pct: bool = False) -> str:
        if v is None:
            return "—"
        if pct:
            return f"{v * 100:.1f}%"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = [
        f"# EnterpriseRAG-Bench Report — {report['generated_at']}",
        "",
        f"**Overall Score:** {fmt(report['overall_score'])}  ",
        f"**Questions evaluated:** {report['n_total']}  (errors: {report['n_errors']})",
        "",
        "| Category | n | Recall | Cited Recall | Extra Docs | Correctness | Completeness % | Overall | Latency ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for category, m in report["per_category"].items():
        lines.append(
            f"| {category} | {m['n']} | {fmt(m['document_recall_mean'], pct=True)} "
            f"| {fmt(m['cited_recall_mean'], pct=True)} | {fmt(m['invalid_extra_docs_mean'])} "
            f"| {fmt(m['correctness_rate'], pct=True)} | {fmt(m['completeness_mean_pct'])} "
            f"| {fmt(m['overall_score'])} | {fmt(m.get('mean_latency_ms'))} |"
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")