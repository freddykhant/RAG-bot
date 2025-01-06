import operator 
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document
from rag import retriever, format_docs, rag_prompt, llm, doc_grader_prompt, doc_grader_instructions, llm_json_mode, router_instructions, hallucination_grader_prompt, hallucination_grader_instructions, answer_grader_prompt, answer_grader_instructions
from langchain_core.messages import HumanMessage, SystemMessage
import json

class GraphState(TypedDict):
  question: str
  generation: str
  query: str # binary decision to run general query
  max_retries: int
  answers: int
  loop_step: Annotated[int, operator.add]
  documents: List[str]

def retrieve(state):
  print("\nRETRIEVING DOCUMENTS...\n")
  question = state["question"]

  documents = retriever.invoke(question)
  return{"documents": documents}

def generate(state):
  print("\nGENERATING ANSWERS...\n")
  question = state["question"]
  documents = state["documents"]
  loop_step = state.get("loop_step", 0)

  docs_txt = format_docs(documents)
  rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
  generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
  return {"generation": generation, "loop_step" : loop_step + 1}

def grade_documents(state):
  print("\nCHECKING DOCUMENT RELEVANCE TO THE QUESTION...\n")
  question = state["question"]  
  documents = state["documents"]

  filtered_docs = []
  query = "No"
  for d in documents:
    doc_grader_prompt_formatted = doc_grader_prompt.format(
      document=d.page_content, question=question
    )
    result = llm_json_mode.invoke(
      [SystemMessage(content=doc_grader_instructions)]
      + [HumanMessage(content=doc_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]
    
    # Document is relevant
    if grade.lower() == "yes":
      print("\nDOCUMENT RELEVANT\n")
      filtered_docs.append(d)
    # Document is not relevant
    else:
      print("\nDOCUMENT NOT RELEVANT\n")
      query = "Yes"
      continue
    return {"documents": filtered_docs, "query": query}

### Edges ###

def route_question(state):
  print("\nROUTE QUESTION\n")
  route_question = llm_json_mode.invoke(
    [SystemMessage(content=router_instructions)]
    + [HumanMessage(content=state["question"])]
  )
  source = json.loads(route_question.content)["datasource"]
  if source == "generalinfo":
    print("\nROUTING TO WEB SEARCH\n")
    return "generalinfo"
  elif source == "vectorstore":
    print("\nROUTE QUESTION TO RAG\n")
    return "vectorstore"
  
# def decide_to_generate(state):
#   print("\nASSESS GRADED DOCUMENTS\N")
#   query = state["query"]  

#   if query == "Yes":
#     print("\n Not all documents are relevant to the question, include general query")
#     return "general_query"
#   else:
#     print("\nDecision: Generate\n")
#     return "generate"
  
def grade_generation(state):
  print("CHECK HALLUCINATIONS")
  question = state["question"]
  documents = state["documents"]
  generation = state["generation"]
  max_retries = state.get("max_retries", 3) # default to 3 if not provided

  # Get hallucination grade 
  hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
    documents=format_docs(documents), generation=generation.content
  )
  result = llm_json_mode.invoke(
    [SystemMessage(content=hallucination_grader_instructions)]
    + [HumanMessage(content=hallucination_grader_prompt_formatted)]
  )
  grade = json.loads(result.content)["binary_score"]

  # Check hallucination
  if grade == "yes": 
    print("\nDecision: GENERATION IS GROUNDED IN DOCUMENTS\n")
    # Check question-answering
    print("\nGRADE GENERATION VS QUESTION\n")
    answer_grader_prompt_formatted = answer_grader_prompt.format(
      question=question, generation=generation.content
      )
    result = llm_json_mode.invoke(
      [SystemMessage(content=answer_grader_instructions)]
      + [HumanMessage(content=answer_grader_prompt_formatted)])
    
    grade = json.loads(result.content)["binary_score"]  

    if grade == "yes":
      print("\nDecision: GENERATION ADDRESSES QUESTION\n")
      return "useful"
    elif state["loop_step"] <= max_retries:
      print("\nDecision: GENERATION DOES NOT ADDRESS QUESTION\n")
      return "not useful"
    else:
      print("\nDecision: MAX RETRIES REACHED\n")
      return "max_retries"  
  elif state["loop_step"] <= max_retries:
    print("\nDecision: HALLUCINATION\n")
    return "not supported"
  else:
    print("\nDecision: MAX RETRIES REACHED\n")
    return "max_retries"  
