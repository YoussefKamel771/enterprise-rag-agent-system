from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from controllers import AgentController
from models import ResponseSignal, ProjectModel
from .schemas.agent import AgentAnswerRequest
import logging

logger = logging.getLogger("uvicorn.error")

agent_router = APIRouter(
    prefix="/api/v1/agent",
    tags=["api_v1", "agent"],
)


@agent_router.post("/answer/{project_id}")
async def agent_answer(request: Request, project_id: int, agent_request: AgentAnswerRequest):

    project_model = await ProjectModel.create_instance(db_client=request.app.state.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    agent_controller = AgentController(rag_graph=request.app.state.rag_graph)

    try:
        result = await agent_controller.answer_question(
            project=project,
            query=agent_request.text,
            thread_id=agent_request.thread_id,
        )
    except Exception:
        logger.exception("Agent graph invocation failed for project_id=%s", project_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.AGENT_ANSWER_ERROR.value},
        )

    answer = result.get("final_answer") or result.get("draft_answer")
    if not answer:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.AGENT_ANSWER_ERROR.value,
                "error": result.get("error"),
            },
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.AGENT_ANSWER_SUCCESS.value,
            "answer": answer,
            "citations": result.get("citations", []),
            "question_type": result.get("question_type"),
            "active_agent": result.get("active_agent"),
            "iterations_used": result.get("iteration_count", 0),
            "retrieved_chunks": [r.dict() for r in result.get("retrieved_chunks") or []]
        } 
    )