from ..VectorDBInterface import VectorDBInterface, MetadataFilter
from ..VectorDBEnums import (DistanceMethodEnums, PgVectorTableSchemeEnums, 
                             PgVectorDistanceMethodEnums, PgVectorIndexTypeEnums)
import logging
from typing import List, Optional, Tuple
from models.db_schemas import RetrievedDocument, RetrievalResult
from sqlalchemy.sql import text as sql_text
import json, re, asyncio

class PGVectorProvider(VectorDBInterface):

    def __init__(self, db_client, default_vector_size: int = 786,
                       distance_method: str = None, index_threshold: int=100,
                       fts_language: str = "english", rrf_k: int = 60):
        
        self.db_client = db_client
        self.default_vector_size = default_vector_size
        
        self.index_threshold = index_threshold

        if distance_method == DistanceMethodEnums.COSINE.value:
            distance_method = PgVectorDistanceMethodEnums.COSINE.value
        elif distance_method == DistanceMethodEnums.DOT.value:
            distance_method = PgVectorDistanceMethodEnums.DOT.value

        self.pgvector_table_prefix = PgVectorTableSchemeEnums._PREFIX.value
        self.distance_method = distance_method
        
        # Lexical (full-text) search config — Postgres's own tsvector/ts_rank_cd
        self.fts_language = fts_language
        self.rrf_k = rrf_k

        self.logger = logging.getLogger("uvicorn")
        self.default_index_name = lambda collection_name: f"{collection_name}_vector_idx"
        self.default_fts_index_name = lambda collection_name: f"{collection_name}_fts_idx"

    async def connect(self):
        async with self.db_client() as session:
            async with session.begin():
                await session.execute(sql_text(
                 "CREATE EXTENSION IF NOT EXISTS vector"))
                await session.commit()

    async def disconnect(self):
        # No specific disconnect logic needed for PGVectorProvider
        pass

    async def is_collection_exists(self, collection_name: str) -> bool:

        record = None
        async with self.db_client() as session:
            async with session.begin():
                list_of_tables = sql_text(f'SELECT * FROM pg_tables WHERE tablename = :collection_name')

                results = await session.execute(list_of_tables, {"collection_name": collection_name})
                record = results.scalar_one_or_none()

        return record is not None

    async def list_all_collections(self) -> List:
        records = []
        async with self.db_client() as session:
            async with session.begin():
                list_tbl = sql_text('SELECT tablename FROM pg_tables WHERE tablename LIKE :prefix')
                results = await session.execute(list_tbl, {"prefix": self.pgvector_table_prefix})
                records = results.scalars().all()
        
        return records

    async def get_collection_info(self, collection_name: str) -> dict:
        async with self.db_client() as session:
            async with session.begin():
                
                table_info_sql = sql_text(f'''
                    SELECT schemaname, tablename, tableowner, tablespace, hasindexes 
                    FROM pg_tables 
                    WHERE tablename = :collection_name
                ''')

                count_sql = sql_text(f'SELECT COUNT(*) FROM {collection_name}')

                table_info = await session.execute(table_info_sql, {"collection_name": collection_name})
                record_count = await session.execute(count_sql)

                table_data = table_info.fetchone()
                if not table_data:
                    return None
                
                return {
                    "table_info": {
                        "schemaname": table_data[0],
                        "tablename": table_data[1],
                        "tableowner": table_data[2],
                        "tablespace": table_data[3],
                        "hasindexes": table_data[4],
                    },
                    "record_count": record_count.scalar_one(),
                }

    async def delete_collection(self, collection_name: str):

        if not await self.is_collection_exists(collection_name=collection_name):
            self.logger.warning(f"Collection {collection_name} does not exist.")
            return False
        
        async with self.db_client() as session:
            async with session.begin():
                self.logger.info(f"Deleting collection: {collection_name}")

                delete_sql = sql_text(f'DROP TABLE IF EXISTS {collection_name}')
                await session.execute(delete_sql)
                await session.commit()
        
        return True

    async def create_collection(self, collection_name: str,
                                      embedding_size: int,
                                      do_reset: bool = False):
        
        if do_reset:
            _ = await self.delete_collection(collection_name=collection_name)

        is_collection_exists = await self.is_collection_exists(collection_name=collection_name)
        if not is_collection_exists:
            self.logger.info(f"Creating collection: {collection_name}")
            async with self.db_client() as session:
                async with session.begin():
                    create_sql = sql_text(
                        f'CREATE TABLE {collection_name} ('
                            f'{PgVectorTableSchemeEnums.ID.value} bigserial PRIMARY KEY,'
                            f'{PgVectorTableSchemeEnums.TEXT.value} text, '
                            f'{PgVectorTableSchemeEnums.VECTOR.value} vector({embedding_size}), '
                            f'{PgVectorTableSchemeEnums.METADATA.value} jsonb DEFAULT \'{{}}\', '
                            f'{PgVectorTableSchemeEnums.CHUNK_ID.value} integer, '
                            f'{PgVectorTableSchemeEnums.TEXT_SEARCH.value} tsvector '
                            f"GENERATED ALWAYS AS (to_tsvector('{self.fts_language}', "
                            f'coalesce({PgVectorTableSchemeEnums.TEXT.value}, \'\'))) STORED, '
                            f'FOREIGN KEY ({PgVectorTableSchemeEnums.CHUNK_ID.value}) REFERENCES chunks(chunk_id)'
                        ')'
                    )
                    await session.execute(create_sql)
                    await session.commit()
            
            await self.create_fts_index(collection_name=collection_name)
            return True

        return False
    
    # ------------------------------------------------------------------
    # Full-text search (lexical) index
    # ------------------------------------------------------------------

    async def is_fts_index_existed(self, collection_name: str) -> bool:
        index_name = self.default_fts_index_name(collection_name)
        async with self.db_client() as session:
            async with session.begin():
                check_sql = sql_text(f"""
                                    SELECT 1
                                    FROM pg_indexes
                                    WHERE tablename = :collection_name
                                    AND indexname = :index_name
                                    """)
                results = await session.execute(check_sql, {"index_name": index_name, "collection_name": collection_name})

                return bool(results.scalar_one_or_none())

    async def create_fts_index(self, collection_name: str):
        if await self.is_fts_index_existed(collection_name=collection_name):
            return False

        async with self.db_client() as session:
            async with session.begin():
                self.logger.info(f"Creating GIN full-text index on: {collection_name}")
                index_name = self.default_fts_index_name(collection_name)
                create_idx_sql = sql_text(
                    f'CREATE INDEX {index_name} ON {collection_name} '
                    f'USING GIN ({PgVectorTableSchemeEnums.TEXT_SEARCH.value})'
                )
                await session.execute(create_idx_sql)

        return True

    async def ensure_hybrid_columns(self, collection_name: str):
        """
        Migration helper: backfills the `text_search` generated column and its
        GIN index onto a table that was created before hybrid search was added,
        without touching existing rows or requiring re-embedding. Safe to call
        on a table that already has both — it's a no-op then.
        """
        if not await self.is_collection_exists(collection_name=collection_name):
            self.logger.warning(f"Collection {collection_name} does not exist; nothing to migrate.")
            return False

        async with self.db_client() as session:
            async with session.begin():
                await session.execute(sql_text(
                    f'ALTER TABLE {collection_name} '
                    f'ADD COLUMN IF NOT EXISTS {PgVectorTableSchemeEnums.TEXT_SEARCH.value} tsvector '
                    f"GENERATED ALWAYS AS (to_tsvector('{self.fts_language}', "
                    f'coalesce({PgVectorTableSchemeEnums.TEXT.value}, \'\'))) STORED'
                ))

        await self.create_fts_index(collection_name=collection_name)
        self.logger.info(f"Hybrid search columns/index ensured on: {collection_name}")
        return True
    
    def _build_or_tsquery(self, text: str) -> str:
        """
        Turn a natural-language query into an OR-of-terms tsquery string
        (term1 | term2 | term3 | ...) instead of relying on plainto_tsquery's
        default AND-everything semantics. AND-ing every lexeme in a full
        question almost never matches a document chunk; OR lets partial term
        overlap still surface a lexical hit, closer to how BM25-style
        matching actually behaves. Stemming/stopword handling still happens
        inside Postgres via to_tsquery's text search dictionary — we're only
        changing the operator between terms, not reimplementing normalization.
        """
        words = re.findall(r"\w+", text.lower())
        if not words:
            return ""
        return " | ".join(words)

    async def is_index_existed(self, collection_name: str) -> bool:
        index_name = self.default_index_name(collection_name)
        async with self.db_client() as session:
            async with session.begin():
                check_sql = sql_text(f""" 
                                    SELECT 1 
                                    FROM pg_indexes 
                                    WHERE tablename = :collection_name
                                    AND indexname = :index_name
                                    """)
                results = await session.execute(check_sql, {"index_name": index_name, "collection_name": collection_name})
                
                return bool(results.scalar_one_or_none())
            
    async def create_vector_index(self, collection_name: str,
                                        index_type: str = PgVectorIndexTypeEnums.HNSW.value,
                                        force: bool = False,):
        is_index_existed = await self.is_index_existed(collection_name=collection_name)
        
        # Fast path – already exists and we don't want to force
        if not force and is_index_existed:
            return False
        
        async with self.db_client() as session:
            async with session.begin():
                count_sql = sql_text(f'SELECT COUNT(*) FROM {collection_name}')
                result = await session.execute(count_sql)
                records_count = result.scalar_one()

                if not force and records_count < self.index_threshold:
                    self.logger.info(
                        f"Skipping index on {collection_name}: "
                        f"{records_count} < threshold {self.index_threshold}"
                    )
                    return False
                
                self.logger.info(
                    f"{'Rebuilding' if force else 'Creating'} HNSW index "
                    f"on {collection_name} ({records_count:,} rows) "
                )
                
                index_name = self.default_index_name(collection_name)
                # Drop existing index if we are forcing a rebuild
                if force:
                    await session.execute(sql_text(f"DROP INDEX IF EXISTS {index_name}"))
                
                create_idx_sql = sql_text(f"""
                            CREATE INDEX {index_name}
                            ON {collection_name}
                            USING hnsw ({PgVectorTableSchemeEnums.VECTOR.value} {self.distance_method})
                            WITH (m = 16, ef_construction = 64)
                        """)

                await session.execute(create_idx_sql)
                await session.commit()

                self.logger.info(f"END: Created vector index ({index_name}) for collection: {collection_name}")
                
        return True

    async def reset_vector_index(self, collection_name: str, 
                                       index_type: str = PgVectorIndexTypeEnums.HNSW.value) -> bool:
        
        return await self.create_vector_index(
            collection_name=collection_name,
            index_type=index_type,
            force=True,
        )

    async def insert_one(self, collection_name: str, text: str, vector: list,
                            metadata: dict = None,
                            record_id: str = None):
        
        is_collection_exists = await self.is_collection_exists(collection_name=collection_name)
        if not is_collection_exists:
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False
        
        if not record_id:
            self.logger.error(f"Can not insert new record without chunk_id: {collection_name}")
            return False
        
        async with self.db_client() as session:
            async with session.begin():
                insert_sql = sql_text(f'INSERT INTO {collection_name} '
                                      f'({PgVectorTableSchemeEnums.TEXT.value}, {PgVectorTableSchemeEnums.VECTOR.value}, {PgVectorTableSchemeEnums.METADATA.value}, {PgVectorTableSchemeEnums.CHUNK_ID.value}) '
                                      'VALUES (:text, :vector, :metadata, :chunk_id)'
                                      )
                
                metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata is not None else "{}"
                await session.execute(insert_sql, {
                    'text': text,
                    'vector': "[" + ",".join([ str(v) for v in vector ]) + "]",
                    'metadata': metadata_json,
                    'chunk_id': record_id
                })
                await session.commit()

                await self.create_vector_index(collection_name=collection_name)
        
        return True
    

    async def insert_many(self, collection_name: str, texts: list,
                         vectors: list, metadata: list = None,
                         record_ids: list = None, batch_size: int = 64, create_index_after: bool = True):
        
        is_collection_exists = await self.is_collection_exists(collection_name=collection_name)
        if not is_collection_exists:
            self.logger.error(f"Can not insert new records to non-existed collection: {collection_name}")
            return False
        
        if len(vectors) != len(record_ids):
            self.logger.error(f"Invalid data items for collection: {collection_name}")
            return False
        
        if not metadata or len(metadata) == 0:
            metadata = [None] * len(texts)
        
        async with self.db_client() as session:
            async with session.begin():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_vectors = vectors[i:i + batch_size]
                    batch_metadata = metadata[i:i + batch_size]
                    batch_record_ids = record_ids[i:i + batch_size]

                    values = []

                    for _text, _vector, _metadata, _record_id in zip(batch_texts, batch_vectors, batch_metadata, batch_record_ids):
                        
                        metadata_json = json.dumps(_metadata, ensure_ascii=False) if _metadata is not None else "{}"
                        values.append({
                            'text': _text,
                            'vector': "[" + ",".join([ str(v) for v in _vector ]) + "]",
                            'metadata': metadata_json,
                            'chunk_id': _record_id
                        })
                    
                    batch_insert_sql = sql_text(f'INSERT INTO {collection_name} '
                                    f'({PgVectorTableSchemeEnums.TEXT.value}, '
                                    f'{PgVectorTableSchemeEnums.VECTOR.value}, '
                                    f'{PgVectorTableSchemeEnums.METADATA.value}, '
                                    f'{PgVectorTableSchemeEnums.CHUNK_ID.value}) '
                                    f'VALUES (:text, :vector, :metadata, :chunk_id)')
                    
                    await session.execute(batch_insert_sql, values)

        if create_index_after:
            await self.create_vector_index(collection_name=collection_name)

        return True
                
    async def search_by_vector(self, collection_name: str, query_vector: list, limit: int,
                            offset: int = 0, filters: Optional[MetadataFilter] = None,
                            exclude_chunk_ids: Optional[List[int]] = None) -> List[dict]:
        """
        Pure dense (pgvector cosine) leg of hybrid search. Returns a list of
        dicts — {chunk_id, text, score, rank} — ordered best-first, rank
        starting at 1. Kept separate from search_by_vector because callers
        of this leg (search_hybrid) need chunk_id and rank for fusion, not
        just RetrievedDocument objects.
        """
        T = PgVectorTableSchemeEnums
        vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"
        clauses, params = self._build_filter_clauses(filters, exclude_chunk_ids)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        
        dense_sql = sql_text(
            f'SELECT v.{T.CHUNK_ID.value} AS chunk_id, v.{T.TEXT.value} AS text, '
            f'v.{T.METADATA.value} AS metadata, c.chunk_asset_id AS doc_id, '
            f'a.asset_source_type AS source_type, '
            f'1 - (v.{T.VECTOR.value} <=> :vector) AS score '
            f'FROM {collection_name} v{self._joins()}{where} '
            # ORDER BY the raw distance expression (not the `score` alias) --
            # that is the only form Postgres can serve from the HNSW index.
            f'ORDER BY v.{T.VECTOR.value} <=> :vector '
            f'LIMIT :limit OFFSET :offset'
        )
        params.update({"vector": vector_str, "limit": limit, "offset": offset})

        async with self.db_client() as session:
            async with session.begin():
                if clauses:
                    # HNSW filters *after* the index scan, so a selective filter can
                    # return fewer than `limit` rows. Iterative scan (pgvector>=0.8)
                    # keeps scanning until enough rows pass the filter.
                    await session.execute(sql_text(
                        "SELECT set_config('hnsw.iterative_scan', 'strict_order', true)"))
                rows = (await session.execute(dense_sql, params)).fetchall()

        return [self._leg_row(r, rank) for rank, r in enumerate(rows, start=offset + 1)]

    async def search_lexical(self, collection_name: str, query_text: str, limit: int,
                              offset: int = 0, filters: Optional[MetadataFilter] = None,
                              exclude_chunk_ids: Optional[List[int]] = None) -> List[dict]:
        """
        Pure lexical (Postgres full-text) leg of hybrid search, using an
        OR-of-terms tsquery so partial term overlap still surfaces a match
        (plainto_tsquery's default AND semantics rarely matches a full
        question against a chunk). Returns a list of dicts —
        {chunk_id, text, score, rank} — ordered best-first, rank starting at 1.
        Returns [] (not an error) when the query has no usable terms or
        nothing matches — callers should treat an empty lexical leg as a
        signal worth logging, not silently ignoring.
        """
        
        T = PgVectorTableSchemeEnums
        or_tsquery = self._build_or_tsquery(query_text)
        if not or_tsquery:
            return []
        
        clauses, params = self._build_filter_clauses(filters, exclude_chunk_ids)
        clauses.insert(0, f"v.{T.TEXT_SEARCH.value} @@ to_tsquery('{self.fts_language}', :query)")

        lexical_sql = sql_text(
            f'SELECT v.{T.CHUNK_ID.value} AS chunk_id, v.{T.TEXT.value} AS text, '
            f'v.{T.METADATA.value} AS metadata, c.chunk_asset_id AS doc_id, '
            f'a.asset_source_type AS source_type, '
            f"ts_rank_cd(v.{T.TEXT_SEARCH.value}, to_tsquery('{self.fts_language}', :query), 32) AS score "
            f'FROM {collection_name} v{self._joins()} '
            f"WHERE {' AND '.join(clauses)} "
            f'ORDER BY score DESC LIMIT :limit OFFSET :offset'
        )
        params.update({"query": or_tsquery, "limit": limit, "offset": offset})

        async with self.db_client() as session:
            async with session.begin():
                rows = (await session.execute(lexical_sql, params)).fetchall()

        return [self._leg_row(r, rank) for rank, r in enumerate(rows, start=offset + 1)]
        
    def _rrf_fusion(self, dense_rows, lexical_rows, limit):
        """RRF fusion: score(doc) = sum over legs of 1 / (k + rank_in_leg)"""
        rrf_scores, rows_by_chunk_id = {}, {}
        for leg in (dense_rows, lexical_rows):
            for row in leg:
                cid = row["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + row["rank"])
                rows_by_chunk_id.setdefault(cid, row)

        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        
        return [
            RetrievedDocument(
                chunk_id=cid,
                doc_id=rows_by_chunk_id[cid]["metadata"].get("doc_id", str(cid)),
                source_type=rows_by_chunk_id[cid]["metadata"].get("source_type", str(cid)),
                text=rows_by_chunk_id[cid]["text"],
                score=score,
                rank=rank,
                metadata=rows_by_chunk_id[cid]["metadata"],
            )
            for rank, (cid, score) in enumerate(fused, start=1)
        ]

    async def search_hybrid(self, collection_name: str, query_text: str, query_vector: list,
                            limit: int = 10, offset: int = 0,
                            filters: Optional[MetadataFilter] = None,
                            exclude_chunk_ids: Optional[List[int]] = None,
                            return_debug: bool = False):
        """
        Dense + lexical search, fused via Reciprocal Rank Fusion. 
        
        Returns a List[RetrievedDocument] by default. Pass return_debug=True
        to instead get a dict:
            {
                "results": List[RetrievedDocument],   # fused, final
                "dense": List[dict],                  # raw dense leg, for inspection
                "lexical": List[dict],                # raw lexical leg, for inspection
            }
        """
        is_collection_exists = await self.is_collection_exists(collection_name=collection_name)
        if not is_collection_exists:
            self.logger.error(f"Can not hybrid-search a non-existed collection: {collection_name}")
            return None

        dense_rows, lexical_rows = await asyncio.gather(
            self.search_by_vector(collection_name, query_vector, limit, offset, filters, exclude_chunk_ids),
            self.search_lexical(collection_name, query_text, limit, offset, filters, exclude_chunk_ids),
        )

        if not dense_rows and not lexical_rows:
            self.logger.warning(
                f"No results from either leg on '{collection_name}' "
                f"(filters={filters}, excluded={len(exclude_chunk_ids or [])})")
            return None
        if not lexical_rows:
            self.logger.warning(f"Lexical leg returned 0 matches (query: {query_text!r}) — effectively dense-only.")
        if not dense_rows:
            self.logger.warning("Dense leg returned 0 matches — effectively lexical-only.")

        # RRF fusion
        results = self._rrf_fusion(dense_rows, lexical_rows, limit)
        
        if not results:
            self.logger.warning(
                f"RRF Fusion returns nothing "
            )

        if return_debug:
            return RetrievalResult(documents=results,
                            dense_debug=dense_rows,
                            lexical_debug=lexical_rows)
        
        return RetrievalResult(documents=results)
    
        # ------------------------------------------------------------------
    # Metadata filtering
    # ------------------------------------------------------------------
    # Logical fields backed by real relational columns (reached via the JOINs
    # below) -> works on already-indexed data, no re-embedding needed.
    # (sql expression, compare case-insensitively?)
    # Any other key falls through to the jsonb `metadata` column.
    _RELATIONAL_FIELDS = {
        "source_type": ("a.asset_source_type", True),
        "doc_id": ("c.chunk_asset_id", False),
    }

    def _joins(self) -> str:
        return (f" JOIN chunks c ON c.chunk_id = v.{PgVectorTableSchemeEnums.CHUNK_ID.value}"
                " JOIN assets a ON a.asset_id = c.chunk_asset_id")

    def _build_filter_clauses(self, filters: Optional[MetadataFilter],
                              exclude_chunk_ids: Optional[List[int]]) -> Tuple[List[str], dict]:
        """MetadataFilter + exclusions -> (WHERE clauses, bind params).
        Filter *keys* never get interpolated into SQL: relational fields come from
        the whitelist above, and jsonb keys are passed as bind parameters."""
        clauses: List[str] = []
        params: dict = {}
        counter = 0

        def resolve(key: str):
            nonlocal counter
            counter += 1
            i = counter
            if key in self._RELATIONAL_FIELDS:
                expr, ci = self._RELATIONAL_FIELDS[key]
            else:
                params[f"k{i}"] = key
                expr = f"(v.{PgVectorTableSchemeEnums.METADATA.value} ->> CAST(:k{i} AS text))"
                ci = True
            return i, expr, ci

        if filters:
            for key, value in filters.equals.items():
                i, expr, ci = resolve(key)
                clauses.append(f"{'LOWER(' + expr + ')' if ci else expr} = :v{i}")
                params[f"v{i}"] = str(value).lower() if ci else str(value)

            for key, values in filters.in_.items():
                if not values:
                    continue
                i, expr, ci = resolve(key)
                clauses.append(f"{'LOWER(' + expr + ')' if ci else expr} = ANY(CAST(:v{i} AS text[]))")
                params[f"v{i}"] = [str(v).lower() if ci else str(v) for v in values]

            for op, bag in ((">=", filters.gte), ("<=", filters.lte)):
                for key, value in bag.items():
                    i, expr, _ = resolve(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        # jsonb ->> yields text; cast so 9 < 10 numerically, not lexically
                        clauses.append(f"({expr})::numeric {op} CAST(CAST(:v{i} AS text) AS numeric)")
                        params[f"v{i}"] = str(value)
                    else:
                        clauses.append(f"{expr} {op} :v{i}")     # ISO dates sort lexically
                        params[f"v{i}"] = str(value)

        if exclude_chunk_ids:
            clauses.append(f"v.{PgVectorTableSchemeEnums.CHUNK_ID.value} <> ALL(CAST(:excluded AS int[]))")
            params["excluded"] = [int(x) for x in exclude_chunk_ids]

        return clauses, params

    @staticmethod
    def _leg_row(row, rank: int) -> dict:
        md = row.metadata
        if isinstance(md, str):                       # raw text() queries return jsonb as str
            try:
                md = json.loads(md)
            except ValueError:
                md = {}
        return {"chunk_id": row.chunk_id, "doc_id": row.doc_id, "source_type": row.source_type,
                "text": row.text, "metadata": md or {}, "score": float(row.score), "rank": rank}