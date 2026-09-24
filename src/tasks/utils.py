from controllers import NLPController, DataController
from typing import Set, Optional
 
 
# ----------------------------------------------------------------------------
# Named document subsets. Add new entries here (or extend DataController)
# to support other subsets in the future without touching the task logic.
# Each loader takes **kwargs (from `document_set_kwargs`) and returns a set
# of doc/asset ids to restrict indexing to. `None` means "no restriction".
# ----------------------------------------------------------------------------
def _load_gold_doc_ids(**kwargs) -> Set[str]:
    data_controller = DataController()
    return data_controller.load_gold_doc_ids()
 
 
DOCUMENT_SET_LOADERS = {
    "gold": _load_gold_doc_ids,
    # future sets, e.g.:
    # "silver": _load_silver_doc_ids,
}
 
 
def resolve_doc_ids(document_set: Optional[str], document_set_kwargs: Optional[dict]) -> Optional[Set[str]]:
    """
    Resolve a named document_set (e.g. "gold") into a concrete set of
    doc/asset ids to restrict indexing to. Returns None when document_set
    is None, meaning "index everything".
    """
    if not document_set:
        return None
 
    loader = DOCUMENT_SET_LOADERS.get(document_set)
    if loader is None:
        return None  # unknown document_set, treat as "no restriction"
 
    doc_ids = loader()
    if not doc_ids:
        return None  # empty id set, treat as "no restriction"
 
    return doc_ids