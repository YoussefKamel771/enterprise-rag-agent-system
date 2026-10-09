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

def _not_enabled(what: str):
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND,
                        content={"signal": f"{what}_not_enabled"})
    
@agent_router.post("/answer/{project_id}")
async def agent_answer(request: Request, project_id: int, agent_request: AgentAnswerRequest):
 
    project_model = await ProjectModel.create_instance(db_client=request.app.state.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)
 
    agent_controller = AgentController(
        rag_graph=request.app.state.rag_graph,
        memory_service=getattr(request.app.state, "memory_service", None),
    )
 
    try:
        result, thread_id = await agent_controller.answer_question(
            project=project,
            query=agent_request.text,
            user_id=agent_request.user_id,
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
                "thread_id": thread_id,
                "error": result.get("error"),
            },
        )
 
    return JSONResponse(
        content={
            "signal": ResponseSignal.AGENT_ANSWER_SUCCESS.value,
            "thread_id": thread_id,
            "answer": answer,
            "citations": result.get("citations", []),
            "question_type": result.get("question_type"),
            "standalone_question": result.get("standalone_question"),
            "active_agent": result.get("active_agent"),
            "iterations_used": result.get("iteration_count", 0),
            "retrieved_chunks": [r.dict() for r in result.get("retrieved_chunks") or []],
        }
    )
    
# --------------------------------------------------------------------------
# Short-term memory: inspect / delete a conversation
# --------------------------------------------------------------------------
 
@agent_router.get("/thread/{project_id}/{user_id}/{thread_id}")
async def get_thread(request: Request, project_id: int, user_id: str, thread_id: str):
    if request.app.state.memory.checkpointer is None:
        return _not_enabled("checkpointer")
    key = AgentController.thread_key(project_id, user_id, thread_id)
    snapshot = await request.app.state.rag_graph.aget_state({"configurable": {"thread_id": key}})
    values = snapshot.values or {}
    return {
        "thread_id": thread_id,
        "summary": values.get("conversation_summary"),
        "messages": [
            {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in values.get("messages", [])
        ],
    }
 
 
@agent_router.delete("/thread/{project_id}/{user_id}/{thread_id}")
async def delete_thread(request: Request, project_id: int, user_id: str, thread_id: str):
    cp = request.app.state.memory.checkpointer
    if cp is None:
        return _not_enabled("checkpointer")
    key = AgentController.thread_key(project_id, user_id, thread_id)
    if hasattr(cp, "adelete_thread"):
        await cp.adelete_thread(key)
    else:
        cp.delete_thread(key)
    return {"thread_id": thread_id, "deleted": True}
 
 
# --------------------------------------------------------------------------
# Long-term memory: inspect / delete what is remembered about a user
# --------------------------------------------------------------------------
 
@agent_router.get("/memory/{project_id}/{user_id}")
async def list_memories(request: Request, project_id: int, user_id: str):
    service = getattr(request.app.state, "memory_service", None)
    if service is None:
        return _not_enabled("ltm")
    return {"user_id": user_id, "memories": await service.list_all(project_id, user_id)}
 
 
@agent_router.delete("/memory/{project_id}/{user_id}")
async def delete_memories(request: Request, project_id: int, user_id: str):
    service = getattr(request.app.state, "memory_service", None)
    if service is None:
        return _not_enabled("ltm")
    return {"user_id": user_id, "deleted": await service.delete_all(project_id, user_id)}