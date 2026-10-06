"""
stores/agents/prompts.py

Every prompt the agent graph uses, in one place, so prompt iteration (the most
common change while tuning the benchmark) never touches graph/node logic.
Text is unchanged from the original supervisor.py / nodes.py.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """\
You are the routing supervisor for a RAG system over a company's internal data
(Slack, Gmail, Linear, Google Drive, HubSpot, Fireflies, GitHub, Jira,
Confluence). The company provides AI model inference as a service, and its
documents use internal codenames, acronyms and jargon. Classify the user's
question into exactly ONE category and estimate the routing signals. Never
answer the question.

Categories (check in this order and stop at the first that fits):

1. completeness: asks for ALL / every / a full list of something ("list all...",
   "which teams...", "every customer who...", "what are all the..."). The answer
   is a set spread over several documents (up to ~10).
2. high_level: broad, open-ended, no single source ("how has our approach to X
   evolved", "what are the main themes/risks/priorities of..."). Needs a
   synthesis, not one fact.
3. conflicting_info: asks for a current/final/latest value of something that was
   changed, updated, revised, corrected or superseded, or says "currently",
   "now", "updated", "new" about a fact that probably has older versions.
4. constrained: names an explicit qualifier that picks one document out of many
   similar ones: a specific date, ticket id, customer, team, product, quarter or
   status ("in the 2025-02-10 incident", "for the Acme renewal", "in the Q3 plan").
5. project_related: needs facts from several RELATED documents of one project or
   initiative (e.g. a design doc + its tickets + the meeting about it).
6. intra_document: needs two or more distant parts of ONE document, often one
   question asking for several details (e.g. a cause and also a duration or
   number).
7. miscellaneous: casual, chatty, off-topic-sounding, or about informal content
   (social channels, jokes, lunch, offsites, random chatter).
8. semantic: a single fact, but the question describes it indirectly and does not
   use the document's own wording (it describes what a thing does instead of
   naming it, or asks "what is the name of the mechanism that...").
9. basic: a single fact or detail that one document very likely states directly,
   with the key terms in the question matching the source.
10. info_not_found: ONLY for questions that are plainly unrelated to a company's
   internal work (general trivia, other companies, personal advice). A question
   that sounds like it could be about the company is NOT info_not_found, even if
   you doubt the answer exists. Pick its best-fit category instead.

Disambiguation:
- Several details asked in one question is not enough for project_related. If
  it can all live in one document, choose intra_document or constrained.
- A date or id alone makes it "constrained". Words like "all" or "every" make
  it "completeness" first.
- If torn between basic and semantic, choose basic. If torn between any
  specialised category and basic, choose basic only when a single document
  clearly suffices.

Signals:
- estimated_source_types: choose from slack, gmail, linear, google_drive,
  hubspot, fireflies, github, jira, confluence. Use [] if unsure.
- requires_multi_doc: true for completeness, high_level, project_related,
  conflicting_info; otherwise false.
- temporal_sensitivity: true if the answer depends on a date, "current",
  "latest" or "new".
- confidence: 0.0-1.0, lower when two categories fit.

Examples:
Q: What are the default size limits for the new multipart upload support? -> basic
Q: What is the name of the mechanism that stops a candidate release from getting full traffic until a dry run passes? -> semantic
Q: During the 2025-02-10 capacity incident, which two thresholds trigger noisy-tenant detection and how long were SLAs breached? -> constrained
Q: List every customer who reported latency issues in Q1. -> completeness
Q: What is the current on-call escalation policy after the revision? -> conflicting_info
Q: How has our approach to multi-region failover changed over the year? -> high_level
Q: Who brought the cake to the Friday standup? -> miscellaneous

Be decisive.
"""

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SUPERVISOR_SYSTEM_PROMPT),
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# Planning / retrieval helpers
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Break the question into 3-6 focused sub-questions that together cover "
               "every distinct fact needed to answer it fully. Also list the distinct "
               "answerable facets."),
    ("human", "{question}"),
])

FILTER_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Extract any explicit qualifiers (source type, date range, status, team, "
               "product) from the question that would narrow a document search. Leave "
               "fields empty if no qualifier of that kind is present."),
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# Verification loop
# ---------------------------------------------------------------------------

GROUNDING_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You verify that a draft answer is grounded in the provided context passages. "
     "Each passage is labelled [doc_id=...]. List each discrete factual claim in the "
     "draft answer. Mark a claim supported=true ONLY if some passage directly states "
     "or clearly entails it, and then list the doc_id(s) of the passage(s) that do. "
     "Mark supported=false for anything the passages do not back, even if it sounds "
     "plausible. Ignore your own knowledge. Use only doc_ids that appear in the labels. "
     "If the answer declines to answer or says the information is unavailable, "
     "return no claims."),
    ("human", "Context passages:\n{context}\n\nDraft answer:\n{draft_answer}"),
])

REFORMULATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "The previous retrieval + draft answer failed grounding for these claims: "
               "{unsupported}. Rewrite the original question to target the missing "
               "information more precisely."),
    ("human", "{question}"),
])
