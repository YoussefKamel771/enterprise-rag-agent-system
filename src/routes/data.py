from fastapi import APIRouter,  Request, HTTPException, status
from fastapi.responses import JSONResponse
from models import ResponseSignal, ProjectModel
from .schemas.data import ProcessRequest, PipelineRequest
import logging
from tasks.data_processing import process_assets
from celery.result import AsyncResult
from celery_app import celery_app 
from tasks.process_workflow import build_process_then_index_workflow

logger = logging.getLogger("uvicorn.error")

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)


@data_router.post("/process/{project_id}")
async def process_data(request: Request, project_id: int, process_request: ProcessRequest):
    project_model = await ProjectModel.create_instance(db_client=request.app.state.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value},
        )
    
    kwargs = process_request.model_dump(exclude_none=True)        
    task = process_assets.apply_async(
                            args=[project.project_id],
                                kwargs=kwargs,)
                        
    return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"signal": "Processing_DISPATCHED", "task_id": task.id},
        )


@data_router.get("/process/status/{task_id}")
async def get_processing_status(task_id: str):
    """
    Poll a chunking task's progress/result.
    meta while running: {"processed", "total", "chunks_inserted"}
    result on success: whatever _process_assets returned (the "signal" dict)
    """
    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "state": result.state,  # PENDING, PROGRESS, SUCCESS, FAILURE, RETRY
    }

    if result.state == "PROGRESS":
        response["meta"] = result.info

    elif result.state == "SUCCESS":
        response["result"] = result.result

    elif result.state == "FAILURE":
        # result.info is the exception in FAILURE state
        response["error"] = str(result.info)

    return response


@data_router.delete("/process/{task_id}")
async def cancel_processing_task(task_id: str):
    """
    Revoke a running/pending chunking task. Doesn't interrupt work already
    inside a batch iteration, but stops it from continuing after the
    current batch (and prevents starting if still PENDING).
    """
    result = AsyncResult(task_id, app=celery_app)
    if result.state in ("SUCCESS", "FAILURE"):
        raise HTTPException(status_code=400, detail=f"Task already finished ({result.state})")

    celery_app.control.revoke(task_id, terminate=False)
    return {"task_id": task_id, "revoked": True}

@data_router.post("/process-and-index/{project_id}")
async def process_and_index(request: Request, project_id: int, body: PipelineRequest):
    project_model = await ProjectModel.create_instance(db_client=request.app.state.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value},
        )

    process_kwargs = dict(
        strategy=body.strategy,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
        batch_size=body.batch_size,
        delete_existing=body.delete_existing,
        skip_chunked=body.skip_chunked,
        document_set=body.document_set,
        document_set_kwargs=body.document_set_kwargs,
    )
    index_kwargs = dict(
        do_reset=body.do_reset,
        page_size=body.page_size,
        embedding_batch_size=body.embedding_batch_size,
        document_set=body.document_set,          # keep same subset for indexing too
        document_set_kwargs=body.document_set_kwargs,
    )

    workflow = build_process_then_index_workflow(project.project_id, process_kwargs, index_kwargs)
    async_result = workflow.apply_async()

    # apply_async() on a chain returns the AsyncResult of the LAST task.
    # Walk .parent to also surface each stage's own task_id for polling.
    index_result = async_result
    check_result = index_result.parent
    process_result = check_result.parent if check_result else None

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "signal": "PIPELINE_DISPATCHED",
            "process_task_id": process_result.id if process_result else None,
            "check_task_id": check_result.id if check_result else None,
            "index_task_id": index_result.id,
        },
    )