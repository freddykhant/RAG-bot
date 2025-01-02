from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
import json

# load LLM model
local_llm = "llama3.2:3b"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, output_format="json")

# vector store
files = [
  "data/coffee_heaven_sales.csv",
  "data/tech_emporium_sales.csv",
  "data/green_grocers_sales.csv"
]

# load documents
docs = []
for file in files:
  loader = CSVLoader(file)
  docs += loader.load()

# split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs)

# add to vector database
vectorstore = SKLearnVectorStore.from_documents(
  documents=doc_splits,
  embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
)

# create retriever
k = min(3, len(doc_splits)) # ensure k does not exceed available chunks
retriever = vectorstore.as_retriever(k=k)


router_instructions = """You are an expert at routing a user question to a vectorstore or general query.

The vectorstore contains spreadsheets related to the sales of 3 different businesess.

Use the vectorstore for questions on these topics. For all else, use trained/general information.

Return JSON with ONLY single key, datasource, that is 'generalinfo' or 'vectorstore' depending on the question."""

# test router
test_general = llm_json_mode.invoke(
  [SystemMessage(content=router_instructions)]
  + [HumanMessage(content="What is the capital of France?")])

test_general2 = llm_json_mode.invoke(
  [SystemMessage(content=router_instructions)]
  + [HumanMessage(content="What is the capital of Australia?")])

test_vector = llm_json_mode.invoke(
  [SystemMessage(content=router_instructions)]
  + [HumanMessage(content="What is the total sales for Coffee Heaven?")])

test_vector2 = llm_json_mode.invoke(
  [SystemMessage(content=router_instructions)]
  + [HumanMessage(content="What is the total sales for Tech Emporium?")])

print(
  json.loads(test_general.content),
  json.loads(test_general2.content),
  json.loads(test_vector.content),
  json.loads(test_vector2.content)
)

### Retrieval Grader ###

# doc grader instructions
doc_grader_instructions = """ You are a grader assessing the relevance of a retrieved document to a user question.

If the document contains keywords(s) or semantic meaning related to the question, grade it as relevant."""

doc_grader_prompt = """ Here is the retrieved document. \n\n {document} \n\n Here is the user question: \n\n {question}.

Please carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with ONLY single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some relevant information to the question."""

# test retrieval grader
question = "What is the total sales for Coffee Heaven?"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
doc_grader_prompt_formatted = doc_grader_prompt.format(document=doc_txt, question=question)
result = llm_json_mode.invoke(
  [SystemMessage(content=doc_grader_instructions)]
  + [HumanMessage(content=doc_grader_prompt_formatted)]
)
json.loads(result.content)