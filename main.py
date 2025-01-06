from langgraph.graph import StateGraph
from graph import GraphState, retrieve, generate, grade_documents, route_question, grade_generation
from langgraph.graph import END

# initialize the workflow
workflow = StateGraph(GraphState)

# add nodes
workflow.add_node("generate", generate)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)

# add edges
workflow.set_conditional_entry_point(
  route_question,
  {
    "generate": "generate",
    "vectorstore": "retrieve"
  }
)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_conditional_edges(
  "generate",
  grade_generation,
  {
    "not supported": "generate",
    "useful" : END,
    "not useful": "generate",
    "max retries": END
  }
)

graph = workflow.compile()

inputs = {"question": "Summarise the sales for Coffee Heaven", "max_retries": 3}
for event in graph.stream(inputs, stream_mode="values"):
  print(event)