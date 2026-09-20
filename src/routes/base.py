from fastapi import FastAPI, APIRouter, Depends
from helpers.config import Settings, get_settings
import os

base_router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"],
)


@base_router.get("/")
async def welcome(settings: Settings = Depends(get_settings)):
    app_name = settings.APP_NAME
    app_version = settings.APP_VERSION
    return {"message": f"Healthy {app_name} v{app_version} API"}
