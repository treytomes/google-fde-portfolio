"""
LangGraph tool-calling agent for the GCP docs Q&A assistant.

The agent has two tools:
  - search_docs(query)    — RAG pipeline (retrieval + generation from corpus)
  - answer_direct(query)  — Gemini with no corpus context (general knowledge)

Gemini decides which tool to call based on the query. For questions specific to
Vertex AI / GCP documentation it uses search_docs; for general ML/programming
questions it uses answer_direct.

Usage:
    from agent import build_agent, run_query

    agent = build_agent(client, embeddings, chunks, backend="vertex_ai")
    result = run_query(agent, "What is context caching in Vertex AI?")
    print(result["answer"])
    print(result["tool_used"])
"""

import os
import warnings
from pathlib import Path

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent


AGENT_MODEL = "gemini-2.5-flash"

_SYSTEM_PROMPT = """\
You are a helpful assistant specializing in Google Cloud and Vertex AI.

You have two tools available:
- search_docs: searches the official Vertex AI / GCP documentation corpus. Use this
  for questions about specific GCP features, APIs, pricing, configuration, or anything
  that requires up-to-date or detailed technical documentation.
- answer_direct: answers from your training knowledge without searching. Use this for
  general programming questions, ML concepts, or questions that don't require
  GCP-specific documentation.

Always use search_docs for questions about Vertex AI, Gemini models, Google Cloud
services, or anything where documentation accuracy matters. Use answer_direct only
when the question clearly doesn't require GCP documentation.\
"""


def build_agent(client, embeddings: "np.ndarray", chunks: list[dict],
                backend: str = "gemini_api") -> object:
    """Build and return a LangGraph ReAct agent with search_docs and answer_direct tools."""

    # Import here to avoid circular dependency with generation/retrieval modules
    from retrieval import retrieve
    from generation import generate

    # Capture in closure so tools don't need global state
    _client    = client
    _embeddings = embeddings
    _chunks    = chunks

    @tool
    def search_docs(query: str) -> str:
        """Search the official Vertex AI and GCP documentation corpus to answer a question.
        Use for questions about specific GCP features, APIs, SDK usage, pricing, or configuration."""
        context_chunks = retrieve(query, _client, _embeddings, _chunks, k=5)
        answer = generate(query, context_chunks, _client)
        sources = "\n".join(f"  - {url}" for url in answer["sources"])
        return f"{answer['text']}\n\nSources:\n{sources}"

    @tool
    def answer_direct(query: str) -> str:
        """Answer a question directly from training knowledge, without searching documentation.
        Use for general ML concepts, programming questions, or topics that don't require
        GCP-specific documentation."""
        from google import genai as _genai
        direct_prompt = (
            "Answer the following question concisely and accurately from your training knowledge. "
            f"Question: {query}"
        )
        response = _client.models.generate_content(
            model=AGENT_MODEL,
            contents=direct_prompt,
        )
        return response.text

    project  = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_LOCATION", "us-central1")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if backend == "vertex_ai" and project:
            llm = ChatGoogleGenerativeAI(
                model=AGENT_MODEL,
                google_api_key="unused",
                vertexai=True,
                project=project,
                location=location,
            )
        else:
            api_key = os.environ.get("GEMINI_API_KEY")
            llm = ChatGoogleGenerativeAI(
                model=AGENT_MODEL,
                google_api_key=api_key,
            )

    return create_react_agent(
        llm,
        tools=[search_docs, answer_direct],
        prompt=_SYSTEM_PROMPT,
    )


def run_query(agent, question: str) -> dict:
    """Run a query through the agent and return a structured result."""
    result = agent.invoke({"messages": [("user", question)]})

    # Extract tool calls from the message history
    tools_called = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(tc["name"])

    final_answer = result["messages"][-1].content

    return {
        "question":   question,
        "answer":     final_answer,
        "tools_used": tools_called,
        "tool_used":  tools_called[0] if tools_called else "none",
        "messages":   result["messages"],
    }
