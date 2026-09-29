import logging
from celery import chain
from celery_app import celery_app
from tasks.data_processing import process_assets
from tasks.data_indexing import index_project_task

logger = logging.getLogger("celery.pipeline")

# Chunking signals that mean "there's nothing to index" - if we see one of
# these, the chain should stop here instead of running indexing.
NO_OP_SIGNALS = {"no_assets_found", "document_set_not_found", "nothing_to_chunk"}


@celery_app.task(bind=True, name="tasks.process_workflow.check_chunking_result")
def check_chunking_result(self, chunking_result: dict, project_id: int):
    """
    Sits between process_assets -> index_project_task in the chain.
    Celery automatically passes the previous task's return value as the
    first positional arg here. If chunking didn't actually produce
    anything, raise so Celery stops the chain and never dispatches
    indexing.
    """
    signal = chunking_result.get("signal")
    if signal in NO_OP_SIGNALS:
        logger.warning(f"project={project_id}: chunking signal={signal}, skipping indexing")
        raise RuntimeError(f"Nothing to index (chunking signal={signal})")

    logger.info(f"project={project_id}: chunking signal={signal}, proceeding to indexing")
    return chunking_result


def build_process_then_index_workflow(project_id: int, chunk_kwargs: dict, index_kwargs: dict):
    """
    chunk_kwargs -> process_assets(project_id, **chunk_kwargs)
    index_kwargs -> index_project_task(project_id, **index_kwargs)
    """
    return chain(
        process_assets.s(project_id, **chunk_kwargs),
        check_chunking_result.s(project_id=project_id),
        # .si() = immutable signature: ignore whatever check_chunking_result
        # returns, use only the explicit args given here. Without this,
        # Celery would try to pass the chunking dict as index_project_task's
        # first positional arg (project_id) and blow up.
        index_project_task.si(project_id, **index_kwargs),
    )