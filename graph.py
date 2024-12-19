import operator 
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document
from langgraph.graph import END
from RAG import retriever, format_docs, rag_prompt, llm
from langchain_core.messages import HumanMessage, SystemMessage

class GraphState(TypedDict):
  question: str # user question
  generation: str # LLM generation
  web_search: str # binary decision to run web search
  max_retries: int # number of answers generated
  answers: int 
  loop_step: Annotated[int, operator.add]
  documents: List[str] # list of retrieved documents


def retrieve(state):
  print("Retrieving documents...")
  question = state["question"]

  documents = retriever.invoke(question)
  return{"documents": documents}

def generate(state):
  print("Generating answers...")
  question = state["question"]
  documents = state["documents"]
  loop_step = state.get("loop_step", 0)

  docs_txt = format_docs(documents)
  rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
  generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
  return {"generation": generation, "loop_step" : loop_step + 1}