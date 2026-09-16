"""
scripts/02_chunk_enterprise_rag_assets.py

Chunk previously-ingested EnterpriseRAG-Bench Assets and store the results
as DataChunk records.

IMPORTANT: the Asset table does not store document content (only
asset_size = len(content) is kept). So this script re-derives content by
re-reading the same Parquet file used during ingestion and matching rows
on doc_id == asset_id. If you'd rather not depend on the parquet file being
available at chunk time, store content on the Asset (or a separate table)
during ingestion instead -- see the note at the bottom of this docstring.

Chunking strategies are pluggable via ChunkerFactory. Only "recursive" is
implemented for now (backed by langchain's RecursiveCharacterTextSplitter);
add more BaseChunker subclasses and register them in
ChunkerFactory._strategies as needed.

Requires: pip install langchain-text-splitters

Usage:
    python scripts/02_chunk_enterprise_rag_assets.py \
        --parquet-path data/documents/test.parquet \
        --project-id 1 \
        --strategy recursive \
        --chunk-size 1000 \
        --chunk-overlap 100 \
        --batch-size 500

Optional:
    --delete-existing   # wipe existing chunks for the project before re-chunking
    --skip-chunked       # skip assets that already have at least one chunk
"""

import argparse
import asyncio
import sys, os
from pathlib import Path
from typing import List, Optional

import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Allow running from repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from helpers.config import get_settings
from models import AssetModel, ChunkModel, AssetTypeEnum
from models.db_schemas import DataChunk
from controllers import DataController, ProcessController
from controllers.ChunkingStrategies import ChunkerFactory
from controllers.DataController import clean_text

# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

async def chunk_assets(
    parquet_path: str,
    project_id: int,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    delete_existing: bool = False,
    skip_chunked: bool = False,
):
    settings = get_settings()

    postgres_conn = (
        f"postgresql+asyncpg://"
        f"{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}"
        f"/{settings.POSTGRES_MAIN_DATABASE}"
    )

    engine = create_async_engine(postgres_conn)
    db_client = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    asset_model = await AssetModel.create_instance(db_client=db_client)
    chunk_model = await ChunkModel.create_instance(db_client=db_client)
    data_controller = DataController()
    process_controller = ProcessController(project_id=project_id)

    if delete_existing:
        deleted = await chunk_model.delete_chunks_by_project_id(project_id=project_id)
        print(f"Deleted {deleted:,} existing chunks for project {project_id}")

    assets = await asset_model.get_all_project_assets(
        asset_project_id=project_id,
        asset_type=AssetTypeEnum.FILE.value,
    )

    if not assets:
        print(f"No assets found for project {project_id}. Nothing to chunk.")
        await engine.dispose()
        return

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
        print(f"Skipping already-chunked assets; {len(needed_ids):,} remaining.")

    if not needed_ids:
        print("Nothing left to chunk after applying --skip-chunked.")
        await engine.dispose()
        return


    pbar = tqdm(total=len(needed_ids), desc="Chunking assets")
    total_chunks = 0
    missing_content = 0
    matched_ids = set()

    # Stream the parquet file batch by batch. For each batch, chunk and
    # insert immediately, then let that batch's content be garbage
    # collected before the next one loads. Nothing accumulates across
    # the whole 500k-row run.
    for rows in data_controller.iter_parquet_batches(
        parquet_path=parquet_path, columns=["doc_id", "content"], batch_size=batch_size
    ):
        pending: List[DataChunk] = []
        
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
            
            if not content:
                missing_content += 1
                pbar.update(1)
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
                print(f"\n[WARN] Failed to chunk asset {asset.asset_id}: {e}")
                pbar.update(1)
                continue
            pending.extend(asset_chunks)
            total_chunks += len(asset_chunks)
            pbar.update(1)
        
        if pending:
            try:
                await chunk_model.insert_many_chunks(pending, batch_size=batch_size)
            except Exception as e:
                bad_ids = {c.chunk_asset_id for c in pending}
                print(f"\n[WARN] Batch insert failed for {len(bad_ids)} assets, skipping: {bad_ids}")
                print(f"        Error: {e}")
            # pending goes out of scope at next loop iteration -> collectible

        if matched_ids == needed_ids:
            break  # found everything we needed, no need to read the rest of the file

    pbar.close()
    await engine.dispose()

    unmatched = len(needed_ids) - len(matched_ids)
    print("\nDone.")
    print(f"Chunks inserted:                          {total_chunks:,}")
    print(f"Assets skipped (empty content):           {missing_content:,}")
    print(f"Assets never found in parquet:             {unmatched:,}")
    print(f"Project ID:                                {project_id}")
    print(f"Strategy:                                  {strategy}")


def main():
    parser = argparse.ArgumentParser(
        description="Chunk ingested EnterpriseRAG-Bench assets and store DataChunk rows."
    )
    parser.add_argument(
        "--parquet-path",
        default=os.path.join("EnterpriseRAG-Bench", "data", "documents", "test.parquet"),
        help="Path to the same documents parquet file used during ingestion.",
    )
    parser.add_argument("--project-id", type=int, default=1)
    parser.add_argument(
        "--strategy",
        default="recursive",
        choices=list(ChunkerFactory._strategies.keys()),
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--delete-existing", action="store_true")
    parser.add_argument("--skip-chunked", action="store_true")

    args = parser.parse_args()

    asyncio.run(
        chunk_assets(
            parquet_path=args.parquet_path,
            project_id=args.project_id,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            delete_existing=args.delete_existing,
            skip_chunked=args.skip_chunked,
        )
    )


if __name__ == "__main__":
    main()