import uuid

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_community.retrievers import TavilySearchAPIRetriever
from langgraph.graph import MessagesState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from vector_store import longterm_memory_vs
from prompts import tavily_prompt

from typing import List

def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id

def format_retrieved_tavily_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    document = Document(page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id})
    longterm_memory_vs.add_documents([document])
    return memory

@tool
def search_recall_memories(query:str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    
    user_id = get_user_id(config)
    documents = longterm_memory_vs.similarity_search(query, k=3, filter = {"user_id": user_id})
    return [document.page_content for document in documents]

@tool
def call_tavily(question):
    """Uses Tavily to answer questions the LLM can't answer"""

    retriever = TavilySearchAPIRetriever(k=3)
    llm = ChatOpenAI(model="gpt-4o-mini")

    chain = (
    {"context": retriever | format_retrieved_tavily_docs, "question": RunnablePassthrough()}
    | tavily_prompt
    | llm
    | StrOutputParser()
    )
    response = chain.invoke(question)
    return response