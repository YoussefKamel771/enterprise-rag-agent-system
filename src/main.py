from fastapi import FastAPI
from routes import base, nlp
from stores.llm import LLMProviderFactory
from stores.vectordb import VectorDBProviderFactory
from stores import TemplateParser
from helpers.config import get_settings
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

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

    # generation client
    app.state.generation_client = llm_provider_factory.create(settings.GENERATION_BACKEND)
    app.state.generation_client.set_generation_model(settings.GENERATION_MODEL_ID)

    # embedding client
    # app.state.embedding_client = llm_provider_factory.create(settings.EMBEDDING_BACKEND)
    # app.state.embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID, 
    #                                                settings.EMBEDDING_MODEL_SIZE)

    # Vector DB client
    app.state.vectordb_client = vector_db_provider_factory.create(settings.VECTOR_DB_BACKEND)
    await app.state.vectordb_client.connect()    

    app.state.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )


    yield  # This is where the application "lives" and handles requests
    
    # --- Shutdown Logic ---
    await app.state.db_engine.dispose()
    await app.state.vectordb_client.disconnect()


# 2. Pass the lifespan to the FastAPI constructor
app = FastAPI(lifespan=lifespan)

app.include_router(base.base_router)
app.include_router(nlp.nlp_router)
