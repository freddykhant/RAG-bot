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
  chunk_size=3000, chunk_overlap=500
)
doc_splits = text_splitter.split_documents(docs)

# add to vector database
vectorstore = SKLearnVectorStore.from_documents(
  documents=doc_splits,
  embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
)

# create retriever
k = min(30, len(doc_splits)) # ensure k does not exceed available chunks
retriever = vectorstore.as_retriever(k=k)


router_instructions = """You are an expert at routing a user question to a vectorstore or general query.

The vectorstore contains spreadsheets related to the sales of 3 different businesess.

Use the vectorstore for questions on these topics. For all else, use trained/general information.

Return JSON with ONLY single key, datasource, that is 'generalinfo' or 'vectorstore' depending on the question."""

### Retrieval Grader ###

# doc grader instructions
doc_grader_instructions = """ You are a grader assessing the relevance of a retrieved document to a user question.

The document contains CSV rows representing sales data. You need to check if the document contains any records related to the business in the question."""

doc_grader_prompt = """ Here is the retrieved document. \n\n {document} \n\n Here is the user question: \n\n {question}.

Please carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with ONLY single key - binary_score, that is either 'yes' or 'no' score to indicate whether the document contains at least some relevant information to the question."""

### answer generator ###

# prompt
rag_prompt = """You are an assistant and expert on data analysis. 

Here is the sales data:

{context} 

Question:

{question}

Answer the question above based on the data provided.
Do not skip or merge unrelated rows. 

Answer:"""

# post processing
def format_docs(docs):
  return "\n\n".join([doc.page_content for doc in docs])

### hallucination grader ###

# Hallucination Grader Instructions
hallucination_grader_instructions = """

You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS.

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside of the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Hallucination Grader Prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with ONLY two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

# test hallucination grader

### answer grader ###

answer_grader_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Output format:
- Return a JSON object, with two keys:
  - "binary_score": "yes" or "no"
  - "explanation": A string explaining your reasoning.
- Do not return any extra text outside the JSON.
"""

# test generation
#question = "What is the total amount of sales for Green Grocers for 2024-12-24" 
question = "What is the total quantity of items sold by Coffee Heaven for 2024-12-24"
documents = retriever.invoke(question)
docs_txt = format_docs(documents)
rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
print(generation.content)