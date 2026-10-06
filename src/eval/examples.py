"""
eval/examples.py -- shared loader for LangSmith dataset examples.
"""

from langsmith import Client

from .load_questions import stratified_sample


def load_examples(settings, sample: str = "full", categories: list[str] | None = None) -> list:
    client = Client()
    examples = list(client.list_examples(dataset_name=settings.EVAL_LANGSMITH_DATASET))
    if not examples:
        raise SystemExit("Dataset is empty -- run `python -m eval.upload_dataset` first.")
    if categories:
        examples = [e for e in examples if e.metadata["question_type"] in categories]
    if sample != "full":
        examples = stratified_sample(examples, n=int(sample), key=lambda e: e.metadata["question_type"])
    return examples