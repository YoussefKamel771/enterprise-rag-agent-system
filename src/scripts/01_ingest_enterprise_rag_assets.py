"""
scripts/01_ingest_enterprise_rag_assets.py

Bulk-ingest EnterpriseRAG-Bench documents into mini-rag's Asset table.

This script intentionally does NOT chunk documents.

Usage:
    python scripts/01_ingest_enterprise_rag_assets.py \
        --parquet-path data/documents/test.parquet \
        --project-id 1 \
        --batch-size 500

Optional:
    --skip-existing

Design:
- Streams the Parquet file using pyarrow.iter_batches().
- Creates one Asset per document.
- Stores source_type, title, and doc_id in asset_config.
- Does not create DataChunk records.
- Chunking is handled separately by 02_chunk_enterprise_rag_assets.py.
"""

import argparse
import asyncio
import sys, os
from pathlib import Path
import uuid

import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# Allow running from repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from helpers.config import get_settings
from models import AssetModel, ProjectModel, AssetTypeEnum
from models.db_schemas import Asset
from controllers import DataController

async def ingest_assets(
    parquet_path: str,
    project_id: int,
    batch_size: int,
    skip_existing: bool = False,
):
    settings = get_settings()

    postgres_conn = (
        f"postgresql+asyncpg://"
        f"{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}"
        f"/{settings.POSTGRES_MAIN_DATABASE}"
    )

    engine = create_async_engine(postgres_conn)

    db_client = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    project_model = await ProjectModel.create_instance(
        db_client=db_client
    )

    asset_model = await AssetModel.create_instance(
        db_client=db_client
    )
    data_controller = DataController()

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    total_rows = data_controller.get_parquet_row_count(parquet_path)
    pbar = tqdm(
        total=total_rows,
        desc="Ingesting EnterpriseRAG-Bench assets",
    )

    for rows in data_controller.iter_parquet_batches(
        parquet_path=parquet_path,
        columns=["doc_id", "source_type", "title", "content"],
        batch_size=batch_size,
    ):
        asset_records = []

        for row in rows:
            asset = data_controller.build_enterprise_rag_asset(
                row=row, project_id=project.project_id
            )

            if asset is None:
                skipped_assets += 1
                continue

            if skip_existing:
                existing_asset = await asset_model.get_asset_record(
                    asset_project_id=project.project_id,
                    asset_name=asset.asset_name,
                )
                if existing_asset:
                    skipped_assets += 1
                    continue

            asset_records.append(asset)

        if asset_records:
            created_assets = await asset_model.create_many_assets(assets=asset_records)
            inserted_assets += len(created_assets)

        pbar.update(len(rows))

    pbar.close()
    await engine.dispose()

    print("\nDone.")
    print(f"Inserted assets: {inserted_assets:,}")
    print(f"Skipped assets:  {skipped_assets:,}")
    print(f"Project ID:     {project_id}")




def main():
    parser = argparse.ArgumentParser(
        description="Ingest EnterpriseRAG-Bench documents into the Asset table."
    )
    parser.add_argument(
        "--parquet-path",
        default=os.path.join("EnterpriseRAG-Bench", "data", "documents", "test.parquet"),
        help="Path to documents parquet file.",
    )
    parser.add_argument("--project-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()
    asyncio.run(
        ingest_assets(
            parquet_path=args.parquet_path,
            project_id=args.project_id,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
        )
    )


if __name__ == "__main__":
    main()