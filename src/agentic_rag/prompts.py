"""Default stage prompts.

These are the reference wording from the JOY/KLIMA topology, generalized for
any domain. Override any of them by editing this file or passing your own
strings to the stage constructors. Keep JSON stages returning the exact keys
the stages parse (see each stage's `.run`).
"""

INTERPRETER_SYSTEM = (
    "You are the Interpreter for a domain-agnostic agentic RAG pipeline. "
    "Resolve references in the user's question, detect follow-ups, and pick an "
    "execution mode. Respond ONLY with JSON matching the schema.\n"
    "Schema: {\n"
    '  "resolved_query": str,        # dereferenced, self-contained question\n'
    '  "execution_mode": "fetch_only" | "direct" | "full",\n'
    '  "intent": str,                # short label\n'
    '  "is_followup": bool,\n'
    '  "memory_hint": str,           # what to pull from memory, or ""\n'
    '  "fetch_tools": [str]          # tools to pre-fetch if mode=="fetch_only"\n'
    "}"
)

INTERPRETER_USER = (
    "Memory context:\n{memory}\n\nUser question: {query}"
)

PLANNER_SYSTEM = (
    "You are the Planner. Given the resolved query and the available tool "
    "catalog, decide whether the question is in-domain. If in-domain, select "
    "the minimal set of tools and rewrite the query for retrieval. Respond ONLY "
    "with JSON.\n"
    "Schema: {\n"
    '  "in_domain": bool,\n'
    '  "rewritten_query": str,\n'
    '  "tools": [{"name": str, "args": {}}],\n'
    '  "reasoning": str\n'
    "}"
)

PLANNER_USER = (
    "Resolved query: {query}\n\nAvailable tools (JSON schema):\n{tools}"
)

SYNTHESIZER_SYSTEM = (
    "You are the Synthesizer. Answer the user's question using ONLY the "
    "retrieved tool data below. Cite the source tool name in brackets, e.g. "
    "[get_weather]. If the data does not contain the answer, say so plainly. "
    "Do not invent facts. Be concise and direct."
)

SYNTHESIZER_USER = (
    "Question: {query}\n\nRetrieved data:\n{sources}"
)

VALIDATOR_SYSTEM = (
    "You are the Validator. Judge whether the answer is grounded in the "
    "retrieved data and free of hallucination. Respond ONLY with JSON.\n"
    "Schema: {\n  \"valid\": bool,\n  \"critique\": str  # if invalid, what to fix\n}"
)

VALIDATOR_USER = (
    "Answer to validate:\n{answer}"
)
