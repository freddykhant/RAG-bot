import operator 
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document
from langgraph.graph import END
from RAG import retriever, format_docs, rag_prompt, llm, doc_grader_prompt, doc_grader_instructions, llm_json_mode, web_search_tool
from langchain_core.messages import HumanMessage, SystemMessage
import json

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

def grade_documents(state):
  print("Checking document relevance to the question...")
  question = state["question"]
  documents = state["documents"]
  
  filtered_docs = []  
  web_search = "No"
  for d in documents:
    doc_grader_prompt_formatted = doc_grader_prompt.format(
      document=d.page_content, question=question
    )
    result = llm_json_mode.invoke(
      [SystemMessage(content=doc_grader_instructions)]
      + [HumanMessage(content=doc_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]
    # Document relevant
    if grade.lower() == "yes":
      print("Document relevant")
      filtered_docs.append(d)
    # Document not relevant 
    else:
      print("Document not relevant")
      web_search = "Yes"
      continue
    return {"documents": filtered_docs, "web_search": web_search}
  
def web_search(state):
  print("Running web search...")
  question = state["question"]
  documents = state.get("documents", [])

  # Web search
  docs = web_search_tool.invoke({"query": question})
  web_results = "\n".join([d["content"] for d in docs])
  web_results = Document(page_content=web_results)
  documents.append(web_results)
  return {"documents": documents}

### Edges
