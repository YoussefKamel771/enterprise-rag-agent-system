import asyncio
import gc, os
import logging

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

from celery_app import celery_app, get_setup_utils
from helpers.config import get_settings
from models import AssetModel, ChunkModel, AssetTypeEnum
from models.db_schemas import DataChunk
from controllers import DataController, ProcessController, NLPController
from controllers.DataController import clean_text
from tasks.utils import resolve_doc_ids

logger = logging.getLogger("celery.chunking_tasks")


async def _process_assets(self, parquet_path: str, project_id: int, strategy: str,
                         chunk_size: int, chunk_overlap: int, batch_size: int,
                         delete_existing: bool, skip_chunked: bool,
                         document_set: str = None, document_set_kwargs: dict = None):
    (db_engine, db_client, llm_provider_factory, 
            vectordb_provider_factory,
            generation_client, embedding_client,
            vectordb_client, template_parser) = await get_setup_utils()

    try:
        asset_model = await AssetModel.create_instance(db_client=db_client)
        chunk_model = await ChunkModel.create_instance(db_client=db_client)
        data_controller = DataController()
        process_controller = ProcessController(project_id=project_id)

        if delete_existing:
            nlp = NLPController(embedding_client=embedding_client)
            await vectordb_client.delete_collection(collection_name=nlp.create_collection_name(project_id))
            deleted = await chunk_model.delete_chunks_by_project_id(project_id=project_id)
            logger.info(f"Deleted {deleted} existing chunks for project {project_id}")

        restrict_ids = resolve_doc_ids(document_set, document_set_kwargs)
        
        assets = await asset_model.get_all_project_assets(
            asset_project_id=project_id,
            asset_type=AssetTypeEnum.FILE.value,
            asset_ids=restrict_ids,
        )

        if not assets:
            signal = "document_set_not_found" if restrict_ids is not None else "no_assets_found"
            return {"signal": signal, "chunks_inserted": 0}

        asset_by_id = {a.asset_id: a for a in assets}
        needed_ids = set(asset_by_id.keys())


        if skip_chunked:
            async with db_client() as session:
                stmt = select(DataChunk.chunk_asset_id).where(
                    DataChunk.chunk_project_id == project_id
                )
                result = await session.execute(stmt)
                already_chunked = {row[0] for row in result.all()}
            needed_ids -= already_chunked

        if not needed_ids:
            return {"signal": "nothing_to_chunk", "chunks_inserted": 0}

        total_assets = len(needed_ids)
        total_chunks = 0
        missing_content = 0
        matched_ids = set()
        processed = 0

        # Stream the parquet file batch by batch, exactly like the
        # original script: chunk + insert per batch, then let that
        # batch's rows/pending chunks be collected before the next loads.
        for rows in data_controller.iter_parquet_batches(
            parquet_path=parquet_path, columns=["doc_id", "content"], batch_size=batch_size
        ):
            pending = []

            for row in rows:
                doc_id = row.get("doc_id")
                if doc_id is None:
                    continue
                doc_id = str(doc_id)
                if doc_id not in needed_ids:
                    continue

                matched_ids.add(doc_id)
                content = clean_text(row.get("content") or "")
                asset = asset_by_id[doc_id]
                processed += 1

                if not content:
                    missing_content += 1
                    continue

                try:
                    asset_chunks = process_controller.build_enterprise_rag_chunks(
                        asset=asset,
                        content=content,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        strategy=strategy,
                    )
                except Exception as e:
                    logger.warning(f"Failed to chunk asset {asset.asset_id}: {e}")
                    continue

                pending.extend(asset_chunks)
                total_chunks += len(asset_chunks)

            if pending:
                try:
                    await chunk_model.insert_many_chunks(pending, batch_size=batch_size)
                except Exception as e:
                    bad_ids = {c.chunk_asset_id for c in pending}
                    logger.warning(f"Batch insert failed for {len(bad_ids)} assets: {bad_ids} ({e})")
                del pending
                gc.collect()

            self.update_state(
                state="PROGRESS",
                meta={
                    "processed": processed,
                    "total_assets": total_assets,
                    "chunks_inserted": total_chunks,
                },
            )

            if matched_ids == needed_ids:
                break  # found everything needed, no need to read the rest of the file

        unmatched = len(needed_ids) - len(matched_ids)

        return {
            "signal": "chunking_success",
            "chunks_inserted": total_chunks,
            "assets_skipped_empty_content": missing_content,
            "assets_not_found_in_parquet": unmatched,
            "project_id": project_id,
            "strategy": strategy,
        }

    finally:
        await vectordb_client.disconnect()
        await db_engine.dispose()


@celery_app.task(bind=True, name="tasks.data_processing.process_assets", max_retries=1)
def process_assets(self,  
                    project_id: int,
                    parquet_path: str = os.path.join("EnterpriseRAG-Bench", 
                                                     "data", 
                                                     "documents",
                                                     "test.parquet"),
                    strategy: str = "recursive",
                    chunk_size: int = 500, 
                    chunk_overlap: int = 50,
                    batch_size: int = 500,
                    delete_existing: bool = False,
                    skip_chunked: bool = False,
                    document_set: str = None, 
                    document_set_kwargs: dict = None):
    """
    Runs the EnterpriseRAG-Bench chunking pipeline in a Celery worker
    instead of blocking a terminal/HTTP request for the whole run.
    Poll progress the same way as the indexing task
    (celery_app.AsyncResult(task_id) / GET /index/push/status/{task_id}-style
    endpoint) - meta has {"processed", "total", "chunks_inserted"}.
    """
    try:
        return asyncio.run(
            _process_assets(self, parquet_path, project_id, strategy,
                          chunk_size, chunk_overlap, batch_size,
                          delete_existing, skip_chunked,
                           document_set, document_set_kwargs)
        )
    except Exception as exc:
        logger.exception(f"Chunking task failed for project_id={project_id}")
        raise self.retry(exc=exc, countdown=30)