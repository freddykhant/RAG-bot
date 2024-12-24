from langgraph.graph import StateGraph
## from IPython.display import Image, display
from graph import GraphState, web_search, retrieve, grade_documents, generate, route_question, decide_to_generate, grade_generation
from langgraph.graph import END  

workflow = StateGraph(GraphState)

workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

### Buld Graph ###
workflow.set_conditional_entry_point(
  route_question,
  {
    "websearch": "websearch",
    "vectorstore": "retrieve"
  }
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")  
workflow.add_conditional_edges(
  "grade_documents",
  decide_to_generate,
  {
    "websearch": "websearch",
    "generate": "generate"
  }
)
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
# display(Image(graph.get_graph().draw_mermaid_png()))

inputs = {"question": "What are the types of agent memory?", "max_retries": 3}
for event in graph.stream(inputs, stream_mode="values"):
  print(event)