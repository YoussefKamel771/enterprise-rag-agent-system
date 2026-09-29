from fastapi import FastAPI
from routes import base, nlp, data, agent
from stores.llm import LLMProviderFactory
from stores.vectordb import VectorDBProviderFactory
from stores.reranker import RerankerProviderFactory
from stores.agents import build_rag_graph
from stores import TemplateParser
from helpers.config import get_settings
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from models import ProjectModel
from controllers import NLPController

# 1. Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    settings = get_settings()

    postgres_conn = (
            f"postgresql+asyncpg://"
            f"{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}"
            f"/{settings.POSTGRES_MAIN_DATABASE}"
        )
    # We attach the client to the app state so it's accessible in routes

    app.state.db_engine = create_async_engine(postgres_conn)
    app.state.db_client = sessionmaker(
        app.state.db_engine, class_=AsyncSession, expire_on_commit=False
    )

    llm_provider_factory = LLMProviderFactory(settings)
    vector_db_provider_factory = VectorDBProviderFactory(settings, db_client=app.state.db_client)
    reranker_provider_factory = RerankerProviderFactory(settings)

    # generation client
    app.state.generation_client = llm_provider_factory.create(settings.GENERATION_BACKEND)
    app.state.generation_client.set_generation_model(settings.GENERATION_MODEL_ID)

    # embedding client
    app.state.embedding_client = llm_provider_factory.create(settings.EMBEDDING_BACKEND)
    app.state.embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID, 
                                                   settings.EMBEDDING_MODEL_SIZE)

    # Vector DB client
    app.state.vectordb_client = vector_db_provider_factory.create(settings.VECTOR_DB_BACKEND)
    await app.state.vectordb_client.connect()    
    
    # Reranker client
    app.state.reranker_client = reranker_provider_factory.create(settings.RERANKER_BACKEND)
    app.state.reranker_client.set_reranker_model(settings.RERANKER_MODEL_ID)

    app.state.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )
    
    # Agent graph (LangGraph) -- built once, same as every client above.
    # `build_rag_graph` is the single factory entry point (stores/agents);
    # which compiled graph comes back is a config switch
    # (settings.AGENT_GRAPH_PHASE), the same pattern as GENERATION_BACKEND /
    # VECTOR_DB_BACKEND picking a provider.
    async def get_project(project_id: int):
        project_model = await ProjectModel.create_instance(db_client=app.state.db_client)
        return await project_model.get_project_or_create_one(project_id=project_id)
 
    agent_nlp_controller = NLPController(
        vectordb_client=app.state.vectordb_client,
        generation_client=app.state.generation_client,
        embedding_client=app.state.embedding_client,
        reranker_client=app.state.reranker_client,
        template_parser=app.state.template_parser,
    )
 
    app.state.rag_graph = build_rag_graph(
        settings=settings,
        nlp_controller=agent_nlp_controller,
        generation_client=app.state.generation_client,
        template_parser=app.state.template_parser,
        get_project=get_project,
    )


    yield  # This is where the application "lives" and handles requests
    
    # --- Shutdown Logic ---
    await app.state.db_engine.dispose()
    await app.state.vectordb_client.disconnect()


# 2. Pass the lifespan to the FastAPI constructor
app = FastAPI(lifespan=lifespan)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(agent.agent_router)
