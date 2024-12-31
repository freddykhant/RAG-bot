from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacerTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import SKLearneVectorStore
from langchain_nomic.embeddings import NomicEmbeddings

# load LLM model
local_llm = "llama3.2:1b"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, output_format="json")

# vector store
files = [
  "data/coffee_heaven_sales.csv",
  "data/tech_emporium_sales.csv",
  "data/green_grocers_sales.csv"
]
