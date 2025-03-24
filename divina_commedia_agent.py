import argparse
import getpass
import os
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from functools import partial
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from context_retriever import ContextRetriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from factory import model_factory

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "Sei un giovane ricercatore in letteratura italiana e stai supportando degli studenti dell'universita' di lettere nello studio della Divina Commedia."
     "Sei stato incaricato di aiutare gli studenti a comprendere meglio il testo e a rispondere a domande relative al testo."
     "Spiega il testo con un linguaggio semplice ed accessibile, comprensibile anche a chi non ha studiato il testo."
     "Questo primo messaggio è il contesto che ti fornisce il tuo compito. Non riferire al contesto come al contesto."
     ),
    ("user", "<context>\n{context}</context>"),
    ("placeholder", "{messages}")
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mistral-large-latest", "mistral-small-latest", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                        default="mistral-large-latest")
    parser.add_argument("--context_file_path", type=str,
                        default="commedia.txt")
    parser.add_argument("--vector_store_serialized_path",
                        type=str, default="commedia_vector_store")
    parser.add_argument("--disable_rag", action="store_true",
                        help="Disable the RAG node")
    return parser.parse_args()


# Defition of the state of the agent graph
class State(TypedDict):
    """The Graph State for your Agent System"""
    messages: Annotated[list, add_messages]
    context: Annotated[set, (lambda x, y: x.union(y))]

# Define the functions that represent the different nodes of the graph


def retrieval_node(context_retriever: ContextRetriever, state: State, config=None, out_key="context") -> dict:
    """The node responsible to perform the RAG. This must be run with partial to make sure that the node passed to langgraph has the context_retriever embedded."""
    ret = context_retriever(state.get("messages")[-1].content, k=3)
    return {out_key: set([doc.page_content for doc in ret])}


def agent(state: State, config=None) -> dict:
    """Node responsible to perform the LLM call, with the state including the context retrieved."""
    update = {"messages": [
        (PROMPT_TEMPLATE | llm).invoke(state, config=config)]}
    return update


def build_graph(context_retriever: ContextRetriever, disable_rag: bool) -> StateGraph:
    """Build the graph of the agent."""
    retrieval_router = partial(retrieval_node, context_retriever)

    builder = StateGraph(State)
    builder.add_node("start", lambda state: {})
    if not disable_rag:
        builder.add_node("retrieval_router", retrieval_router)
    builder.add_node("agent", agent)
    # Graph edges
    builder.add_edge(START, "start")
    if not disable_rag:
        builder.add_edge("start", "retrieval_router")
        builder.add_edge("retrieval_router", "agent")
    else:
        builder.add_edge("start", "agent")
    builder.add_edge("agent", END)
    app = builder.compile(checkpointer=MemorySaver())
    return app


if __name__ == "__main__":

    args = parse_args()

    llm = model_factory(model_name=args.model, temperature=0.0)

    context_retriever = ContextRetriever(
        context_file_path=args.context_file_path, vector_store_serialized_path=args.vector_store_serialized_path)

    app = build_graph(context_retriever, args.disable_rag)

    while True:
        user_input = input(">> ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting...")
            break

        response = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": 1}}
        )

        print(response["messages"][-1].content)
