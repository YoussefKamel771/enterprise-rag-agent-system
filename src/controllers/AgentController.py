from .BaseController import BaseController
from models.db_schemas import Project
from stores.agents import run_rag_graph
from helpers.config import get_settings


class AgentController(BaseController):
    """
    Owns nothing heavy itself -- `rag_graph` is the already-compiled graph
    built once in main.py's lifespan (stores/agents.build_rag_graph) and
    injected here, same relationship NLPController has to its vectordb/
    generation/embedding/reranker clients.
    """

    def __init__(self, rag_graph):
        super().__init__()
        self.rag_graph = rag_graph
        self.app_settings = get_settings()

    async def answer_question(self, project: Project, query: str, thread_id: str = None):
        thread_id = thread_id or f"project-{project.project_id}"
        return await run_rag_graph(
            self.rag_graph,
            question=query,
            project_id=project.project_id,
            thread_id=thread_id,
            max_iterations=self.app_settings.AGENT_MAX_ITERATIONS,
            recursion_limit=self.app_settings.AGENT_RECURSION_LIMIT,
        )