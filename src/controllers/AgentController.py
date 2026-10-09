import logging
import uuid

from .BaseController import BaseController
from models.db_schemas import Project
from stores.agents import run_rag_graph
from helpers.config import get_settings
from tasks.memory import write_memory_task

logger = logging.getLogger("uvicorn")


class AgentController(BaseController):
    """
    `rag_graph` is the already-compiled graph built once in main.py's lifespan.

    Thread keys are scoped by project AND user, so two users of one project can
    never share a conversation (the old default, `project-{id}`, would have
    merged everyone into one thread the moment a checkpointer existed).
    """

    def __init__(self, rag_graph, memory_service=None):
        super().__init__()
        self.rag_graph = rag_graph
        self.memory_service = memory_service
        self.app_settings = get_settings()

    @staticmethod
    def thread_key(project_id, user_id, thread_id) -> str:
        return f"{project_id}:{user_id or 'anon'}:{thread_id}"

    async def answer_question(self, project: Project, query: str,
                              user_id: str = None, thread_id: str = None):
        """Returns (final_state, thread_id). A missing thread_id starts a new thread."""
        thread_id = thread_id or uuid.uuid4().hex
        result = await run_rag_graph(
            self.rag_graph,
            question=query,
            project_id=project.project_id,
            thread_id=self.thread_key(project.project_id, user_id, thread_id),
            user_id=user_id,
            max_iterations=self.app_settings.AGENT_MAX_ITERATIONS,
            recursion_limit=self.app_settings.AGENT_RECURSION_LIMIT,
        )
        self._dispatch_memory_write(project, user_id, query, result)
        return result, thread_id

    def _dispatch_memory_write(self, project, user_id, query, result):
        s = self.app_settings
        answer = result.get("final_answer") or result.get("draft_answer")
        # grounding_passed only exists in phase 3; default True elsewhere.
        if not (s.AGENT_LTM_ENABLED and user_id and answer
                and not result.get("error") and result.get("grounding_passed", True)):
            return
        try:
            
            write_memory_task.delay({
                "project_id": project.project_id,
                "user_id": user_id,
                "original_question": query,
                "standalone_question": result.get("standalone_question"),
                "citations": result.get("citations", []),
                "question_type": result.get("question_type"),
            })
        except Exception:
            # A broker outage must never fail the user's request.
            logger.exception("could not dispatch memory write")