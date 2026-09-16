# CustomerSupportRAG

# SupportRAG — Phase 0 Architecture (mini-rag-style)

This supersedes the earlier MVC-labeled design. Instead of `controllers/services/repositories/views` folders, it mirrors your **mini-rag** project's actual conventions: `routes` (thin FastAPI routers returning `JSONResponse`), `controllers` (business logic classes, not DB access), `models` (both Pydantic schemas *and* the async data-access classes), `stores` (pluggable provider factories for LLM/vector DB/templates), `helpers` (settings). Same responsibilities as before, different — and now familiar — folder names.

---

## 1. Mapping: mini-rag concept → SupportRAG Phase 0 role

| mini-rag piece | Reused as-is? | SupportRAG Phase 0 role |
|---|---|---|
| `helpers/config.py` (`Settings`) | Yes | Add `EMBEDDING_MODEL_NAME`, `LLM_PROVIDER`, `DATABASE_URL` (Postgres instead of Mongo URL) |
| `controllers/BaseController.py` | Yes, same shape | Shared `app_settings`, path helpers |
| `controllers/ProjectController.py` | Dropped for Phase 0 | mini-rag scopes files per-project; SupportRAG has no "project" concept yet — could return in a later multi-tenant phase, not needed now |
| `models/db_schemas/*.py` | Yes, same idiom | Pydantic schema classes: `KnowledgeDoc`, `DocChunk`, `Conversation`, `Message`, plus stubs `Ticket`, `FeedbackEvent`, `EvalRun` |
| `models/*.py` (`ProjectModel`, `ChunkModel`, `AssetModel`) | Yes, same idiom, **simplified** | Async data-access classes per schema. mini-rag's `init_collection()` step (Mongo index creation at runtime) isn't needed here — Postgres DDL/indexes live in Alembic migrations instead, so these classes are pure CRUD wrappers, no init step |
| `models/enums/ResponseEnums.py` (`ResponseSignal`) | Yes, directly | Same pattern: every route returns `{"signal": ResponseSignal.X.value}` — keep this, it's a good habit |
| `models/enums/*` (`AssetTypeEnum`, `ProcessingEnum`) | Yes, same idiom | New enums: `DocTypeEnum`, `ChannelEnum`, `MessageRoleEnum`, `TicketStatusEnum` |
| `stores/llm/` (Factory + Interface + providers) | Yes, unchanged | Reuse directly — `LLMProviderFactory`, `LLMInterface`, `OpenAIProvider`/`CoHereProvider` need no changes for Phase 0 |
| `stores/vectordb/` (Factory + Interface + `QdrantDBProvider`) | Yes, add a provider | Add `PGVectorProvider` implementing the existing `VectorDBInterface` — keeps the same pluggable-backend pattern mini-rag already has, even though you're on Postgres now |
| `stores/llm/templates/` + `locales/{en,ar}/` | **Yes — this is the key reuse** | This already solves your multilingual prompt problem. Add new template groups (`chat.py`, `classification.py`, `safety.py`) alongside the existing `rag.py`, each with `en/` and `ar/` versions, parsed the same way via `TemplateParser.get(group, key, vars)` |
| `routes/` + `routes/schemas/` | Yes, same idiom | `base.py`, `chat.py`, `admin.py` routers; `schemas/chat.py`, `schemas/admin.py` request models |
| `main.py` lifespan pattern | Yes, same idiom | Attach `db_engine`/session factory instead of `mongodb_client`; everything else (llm factory, vectordb factory, template parser) is a direct copy of your existing lifespan logic |

---

## 2. Folder Structure

```
src/
├── main.py
├── requirements.txt
├── .env.example
│
├── helpers/
│   ├── __init__.py
│   └── config.py                    # Settings — add DATABASE_URL, EMBEDDING_MODEL_NAME
│
├── controllers/
│   ├── __init__.py
│   ├── BaseController.py            # unchanged pattern: app_settings, path helpers
│   ├── IngestionController.py       # chunk raw text, call embedding client, hand off to models
│   ├── RetrievalController.py       # naive single-vector search orchestration
│   └── GenerationController.py      # build prompt via TemplateParser, call generation_client
│
├── models/
│   ├── __init__.py
│   ├── BaseDataModel.py             # unchanged pattern: db_client + app_settings
│   ├── KnowledgeDocModel.py         # CRUD for knowledge_docs
│   ├── ChunkModel.py                # CRUD + similarity_search for doc_chunks
│   ├── ConversationModel.py
│   ├── MessageModel.py
│   ├── TicketModel.py               # stub, unused until Phase 2/4
│   ├── FeedbackModel.py             # stub, unused until Phase 5
│   ├── EvalRunModel.py              # stub, unused until Phase 6
│   │
│   ├── enums/
│   │   ├── __init__.py
│   │   ├── ResponseEnums.py         # ResponseSignal — same idiom as mini-rag
│   │   ├── DataBaseEnum.py          # table name constants, mirrors mini-rag's collection-name enum
│   │   ├── DocTypeEnum.py           # FAQ | POLICY | MANUAL | TICKET_HISTORY
│   │   ├── ChannelEnum.py           # API | CHAT | EMAIL
│   │   ├── MessageRoleEnum.py       # CUSTOMER | ASSISTANT | AGENT
│   │   └── TicketStatusEnum.py      # OPEN | ESCALATED | RESOLVED
│   │
│   └── db_schemas/
│       ├── __init__.py
│       ├── knowledge_doc.py         # Pydantic schema, same idiom as project.py/asset.py
│       ├── doc_chunk.py             # includes RetrievedDocument-style result schema
│       ├── conversation.py
│       ├── message.py
│       ├── ticket.py
│       ├── feedback_event.py
│       └── eval_run.py
│
├── routes/
│   ├── __init__.py
│   ├── base.py                      # unchanged: GET /api/v1/
│   ├── chat.py                      # POST /api/v1/chat/{conversation_id}
│   ├── admin.py                     # POST /api/v1/admin/ingest
│   └── schemas/
│       ├── __init__.py
│       ├── chat.py                  # ChatRequest, matches ProcessRequest idiom
│       └── admin.py                 # IngestRequest
│
└── stores/
    ├── __init__.py
    ├── llm/                         # unchanged from mini-rag
    │   ├── LLMEnums.py
    │   ├── LLMInterface.py
    │   ├── LLMProviderFactory.py
    │   ├── providers/
    │   │   ├── OpenAIProvider.py
    │   │   └── CoHereProvider.py
    │   └── templates/
    │       ├── template_parser.py
    │       └── locales/
    │           ├── en/
    │           │   ├── rag.py       # existing
    │           │   ├── chat.py      # new: greeting/fallback/abstain templates
    │           │   └── classification.py  # new: intent/priority prompt templates
    │           └── ar/
    │               ├── rag.py       # existing
    │               ├── chat.py
    │               └── classification.py
    │
    └── vectordb/                    # same Factory/Interface pattern, new provider
        ├── VectorDBEnums.py
        ├── VectorDBInterface.py
        ├── VectorDBProviderFactory.py
        └── providers/
            └── PGVectorProvider.py  # new: implements VectorDBInterface using pgvector/SQLAlchemy
```

---

## 3. Key Adaptations from mini-rag → SupportRAG

**1. `models/*Model.py` classes lose the `init_collection()` step.**
Mongo is schemaless, so mini-rag's `ChunkModel.create_instance()` checks/creates indexes at runtime. Postgres tables and indexes (including the `ivfflat` vector index) are created by Alembic migrations instead. So `ChunkModel`, `KnowledgeDocModel`, etc. become simpler — just a thin async wrapper around a SQLAlchemy session, with methods like `insert_many_chunks`, `similarity_search`, `get_by_project` — no `create_instance`/`init_collection` dance needed. If you want to keep the async-factory *pattern* for consistency with the rest of the codebase, that's fine too — just have `create_instance` skip straight to returning the instance.

**2. `stores/vectordb/providers/PGVectorProvider.py` implements the same `VectorDBInterface`** your `QdrantDBProvider` already implements (`connect`, `create_collection`, `insert_many`, `search_by_vector`, etc.), just backed by a `pgvector` column + SQLAlchemy instead of the Qdrant client. This preserves the "swap vector backends via factory" flexibility mini-rag already has, even though Phase 0 only ever uses one backend.

**3. `stores/llm/templates/locales/{en,ar}/` gets new template groups, not new machinery.**
This is the biggest win from copying mini-rag's structure: your multilingual prompt system is already built. Phase 0 only needs a `chat.py` template group (system prompt, citation format, abstain message) in both `en/` and `ar/`, following the exact `Template("\n".join([...]))` pattern your `rag.py` locales already use. `TemplateParser.set_language()` already does the fallback-to-default-language logic you'll want when Arabic content is incomplete.

**4. `controllers/` stays DB-agnostic business logic, same as mini-rag's `ProcessController`/`NLPController`.**
`IngestionController.chunk_and_prepare()` mirrors `ProcessController.process_file_content()` almost exactly (it takes raw text, returns chunk objects) — you can copy that class's shape directly. `RetrievalController`/`GenerationController` mirror `NLPController.search_vector_db_collection()` / `answer_rag_question()` — same two-step "embed query → search → build prompt via template parser → call generation client" flow you already wrote once.

**5. `ResponseSignal` enum pattern is kept and extended.**
Add `KB_INGESTED_SUCCESS`, `CHAT_ANSWER_SUCCESS`, `CHAT_ABSTAINED`, etc. alongside your existing signals — every route keeps returning `{"signal": ResponseSignal.X.value, ...}`, which is a good, consistent convention worth keeping as the project grows.

**6. `main.py` lifespan block is nearly a direct copy.**
Replace `AsyncIOMotorClient` with an async SQLAlchemy engine + sessionmaker attached to `app.state`; keep `llm_provider_factory`, `vectordb_provider_factory` (now resolving to `PGVectorProvider`), and `template_parser` exactly as they are today.

---

## 4. What Changed vs. the Previous (generic-MVC) Design

- No `views/` folder — routes return `JSONResponse` directly with a `signal` field, same as mini-rag, rather than a separate response-shaping layer.
- No `repositories/` folder — the `models/*Model.py` classes *are* the data-access layer (schema + access logic live in `models/`, split between `db_schemas/` for shape and top-level `*Model.py` for behavior), exactly like mini-rag's `AssetModel`/`ChunkModel`/`ProjectModel`.
- `services/` is renamed to `controllers/` to match mini-rag's naming (mini-rag's "controllers" are what most frameworks would call services — business logic, not routing).
- Database schema (tables, columns, pgvector index) from the earlier design doc is unchanged — only the module/folder organization around it has moved.

---

*Next step: want me to scaffold this exact folder tree (empty files with docstrings/class stubs, matching mini-rag's actual file style) so you have something to fill in?*