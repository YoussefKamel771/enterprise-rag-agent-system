"""
eval/storage.py -- Section 6.3

One row per (question_id, phase, run_tag) in `eval_runs` (migration:
models/db_schemas/supportRag/alembic/versions/a1f4c9d2e7b3_add_eval_runs_table.py).
Never overwrites -- history across phases/commits is the point.
"""

import json
import logging

from sqlalchemy import text

logger = logging.getLogger("uvicorn.eval")

INSERT_SQL = text("""
    INSERT INTO eval_runs
        (question_id, question_type, phase, run_tag, retrieved_doc_ids,
         draft_answers, final_answer, judge_verdict, telemetry, routing_trace)
    VALUES
        (:question_id, :question_type, :phase, :run_tag, :retrieved_doc_ids,
         CAST(:draft_answers AS jsonb), :final_answer, CAST(:judge_verdict AS jsonb),
         CAST(:telemetry AS jsonb), CAST(:routing_trace AS jsonb))
""")


async def store_result(db_client, record: dict) -> None:
    retrieved_doc_ids = sorted({
        getattr(d, "doc_id", None) for d in record["final_state"].get("retrieved_chunks", [])
        if getattr(d, "doc_id", None)
    })
    verdict = record.get("judge_verdict")
    verdict_payload = verdict.to_record() if verdict else {"error": record.get("error") or "judge_failed"}

    params = {
        "question_id": record["question_id"],
        "question_type": record["question_type"],
        "phase": record["phase"],
        "run_tag": record["run_tag"],
        "retrieved_doc_ids": retrieved_doc_ids,
        "draft_answers": json.dumps(record.get("draft_answers", []), default=str),
        "final_answer": record.get("generated_answer"),
        "judge_verdict": json.dumps(verdict_payload, default=str),
        "telemetry": json.dumps(record.get("telemetry", {}), default=str),
        "routing_trace": json.dumps(record.get("routing", {}), default=str),
    }

    async with db_client() as session:
        async with session.begin():
            try:
                await session.execute(INSERT_SQL, params)
            except Exception:
                logger.exception("Failed to persist eval_runs row for question_id=%s", record["question_id"])