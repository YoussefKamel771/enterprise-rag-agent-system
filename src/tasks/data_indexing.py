import asyncio
import logging
import gc
from celery import group, chord
from celery.exceptions import SoftTimeLimitExceeded
import time
from celery_app import celery_app, get_setup_utils
from models import ChunkModel, ProjectModel
from models import ResponseSignal
from controllers import NLPController

logger = logging.getLogger("celery_task")


async def _run_indexing(self, project_id: int, do_reset: bool,
                         page_size: int, embedding_batch_size: int):
    (db_engine, db_client, llm_provider_factory, 
        vectordb_provider_factory,
        generation_client, embedding_client,
        vectordb_client, template_parser) = await get_setup_utils()
    
    start_time = time.perf_counter()
  
    try:
        project_model = await ProjectModel.create_instance(db_client=db_client)
        chunk_model = await ChunkModel.create_instance(db_client=db_client)
 
        project = await project_model.get_project_or_create_one(project_id=project_id)
        if not project:
            return {"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value}
 
        nlp_controller = NLPController(
            vectordb_client=vectordb_client,
            generation_client=generation_client,
            embedding_client=embedding_client,
            template_parser=None,  # not needed for indexing
        )
 
        collection_name = nlp_controller.create_collection_name(project_id=project.project_id)
 
        await vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=embedding_client.embedding_size,
            do_reset=do_reset,
        )
 
        total_chunks_count = await chunk_model.get_total_chunks_count(project_id=project.project_id)
 
        has_records = True
        page_no = 1
        inserted_items_count = 0
 
        while has_records:
            # Only ONE page of chunks (default 100 rows) is ever resident
            # in memory at a time - never the whole project.
            page_chunks = await chunk_model.get_project_chunks(
                project_id=project.project_id, page_no=page_no, page_size=page_size,
            )
 
            if not page_chunks:
                has_records = False
                break
 
            page_no += 1
            chunks_ids = [c.chunk_id for c in page_chunks]
 
            is_inserted = await nlp_controller.index_into_vector_db(
                project=project,
                chunks=page_chunks,
                chunks_ids=chunks_ids,
                do_reset=False,               # collection already created above
                create_index_after=False,            # defer the ANN index build to the end
                embedding_batch_size=embedding_batch_size,
            )
 
            if not is_inserted:
                return {"signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value}
 
            inserted_items_count += len(page_chunks)
 
            # Drop references to this page's data before pulling the next
            # one, so the interpreter can actually free it rather than
            # quietly letting pages 1..N-1 stay reachable via local scope.
            del page_chunks, chunks_ids
            gc.collect()
            
            elapsed_so_far = round(time.perf_counter() - start_time, 2)
 
            if total_chunks_count:
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "inserted": inserted_items_count,
                          "total": total_chunks_count,
                          "progress": f"{round(inserted_items_count / total_chunks_count * 100, 2)}%",
                          "elapsed_seconds": f"{elapsed_so_far}s"},
                    
                )
 
        # Time specific heavy operations separately if needed
        index_build_start = time.perf_counter()
        await vectordb_client.create_vector_index(
            collection_name=collection_name, force=True
        )
        index_build_duration = round(time.perf_counter() - index_build_start, 2)

        # 3. Calculate final total elapsed time
        total_duration = round(time.perf_counter() - start_time, 2)

        logger.info(
            f"Indexing completed in {total_duration}s (Index creation: {index_build_duration}s)"
        )
 
        return {
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count,
            "execution_time_seconds": total_duration,
            "index_build_time_seconds": index_build_duration,
        }
 
    finally:
        await vectordb_client.disconnect()
        await db_engine.dispose()
 
 
@celery_app.task(bind=True, name="tasks.nlp_tasks.index_project_task", max_retries=2)
def index_project_task(self, project_id: int, do_reset: bool = False,
                        page_size: int = 1000, embedding_batch_size: int = 64):
    """
    Runs the heavy vector-indexing job in a Celery worker instead of inside
    the request/response cycle, so:
      - the API returns instantly with a task_id instead of blocking/timing
        out on a long HTTP request
      - the work happens in a process that gets recycled after the job
        (worker_max_tasks_per_child=1), so memory doesn't creep up across runs
      - only one page of chunks is ever held in memory at once
    """
    try:
        return asyncio.run(
            _run_indexing(self, project_id, do_reset, page_size, embedding_batch_size)
        )
    except Exception as exc:
        logger.exception("Indexing task failed for project_id=%s", project_id)
        raise self.retry(exc=exc, countdown=30)
 