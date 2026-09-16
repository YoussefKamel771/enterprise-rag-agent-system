from typing import List
import uuid
from sqlalchemy.dialects.postgresql import insert
from .BaseDataModel import BaseDataModel
from .enums.DataBaseEnum import DataBaseEnum
from .db_schemas import Asset
from bson import ObjectId
from sqlalchemy.future import select

class AssetModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.db_client = db_client

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        return instance

    async def create_asset(self, asset: Asset):

        async with self.db_client() as session:
            async with session.begin():
                session.add(asset)
            await session.refresh(asset)
        return asset
    
    async def create_many_assets(self, assets: List[Asset]) -> List[Asset]:
        if not assets:
            return []

        async with self.db_client() as session:

            values = [
                {
                    "asset_id": asset.asset_id,
                    "asset_uuid": asset.asset_uuid or uuid.uuid4(),
                    "asset_uuid": asset.asset_uuid,
                    "asset_source_type": asset.asset_source_type,
                    "asset_name": asset.asset_name,
                    "asset_size": asset.asset_size,
                    "asset_type": asset.asset_type,
                    "asset_config": asset.asset_config,
                    "asset_project_id": asset.asset_project_id,
                }
                for asset in assets
            ]

            stmt = insert(Asset).values(values)

            stmt = stmt.on_conflict_do_nothing(
                index_elements=[Asset.asset_id]
            )

            await session.execute(stmt)
            await session.commit()

        return assets

    async def get_all_project_assets(self, asset_project_id: int, asset_type: str, batch_size: int = 2000):
        async with self.db_client() as session:
            stmt = select(Asset).where(
                Asset.asset_project_id == asset_project_id,
                Asset.asset_type == asset_type
            ).execution_options(yield_per=batch_size)

            result = await session.stream(stmt)
            records = [row[0] async for row in result]
        return records

    async def get_asset_record(self, asset_project_id: int, asset_name: str):

        async with self.db_client() as session:
            stmt = select(Asset).where(
                Asset.asset_project_id == asset_project_id,
                Asset.asset_name == asset_name
            )
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()
        return record
