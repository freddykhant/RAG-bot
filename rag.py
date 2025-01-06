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
# test_general = llm_json_mode.invoke(
#   [SystemMessage(content=router_instructions)]
#   + [HumanMessage(content="What is the capital of France?")])

# test_general2 = llm_json_mode.invoke(
#   [SystemMessage(content=router_instructions)]
#   + [HumanMessage(content="What is the capital of Australia?")])

# test_vector = llm_json_mode.invoke(
#   [SystemMessage(content=router_instructions)]
#   + [HumanMessage(content="What is the total sales for Coffee Heaven?")])

# test_vector2 = llm_json_mode.invoke(
#   [SystemMessage(content=router_instructions)]
#   + [HumanMessage(content="What is the total sales for Tech Emporium?")])

# print(
#   json.loads(test_general.content),
#   json.loads(test_general2.content),
#   json.loads(test_vector.content),
#   json.loads(test_vector2.content)
# )

### Retrieval Grader ###

# doc grader instructions
doc_grader_instructions = """ You are a grader assessing the relevance of a retrieved document to a user question.

The document contains CSV rows representing sales data. You need to check if the document contains any records related to the business in the question."""

doc_grader_prompt = """ Here is the retrieved document. \n\n {document} \n\n Here is the user question: \n\n {question}.

Please carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with ONLY single key - binary_score, that is either 'yes' or 'no' score to indicate whether the document contains at least some relevant information to the question."""

# test retrieval grader
# question = "What is the total sales for Coffee Heaven?"
# docs = retriever.invoke(question)
# doc_txt = docs[1].page_content
# doc_grader_prompt_formatted = doc_grader_prompt.format(
#     document=doc_txt, question=question
# )
# result = llm_json_mode.invoke(
#     [SystemMessage(content=doc_grader_instructions)]
#     + [HumanMessage(content=doc_grader_prompt_formatted)]
# )
# json.loads(result.content)
# print(result.content)

### answer generator ###

# prompt
rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to these questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""

# post processing
def format_docs(docs):
  return "\n\n".join([doc.page_content for doc in docs])

# test generation
question = "What are the TOTAL sales for Coffee Heaven for 2024-12-24?"
docs = retriever.invoke(question)
docs_txt = format_docs(docs)
rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
print("\n")
print(generation.content)

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

# hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
#     documents=docs_txt, generation=generation.content
# )
# result = llm_json_mode.invoke(
#    [SystemMessage(content=hallucination_grader_instructions)] 
#    + [HumanMessage(content=hallucination_grader_prompt_formatted)]
# )
# json.loads(result.content)

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
- Do not return any extra text outside the JSON."""

# test answer grader
# question = "What is the total sales for Coffee Heaven?"
# answer = "The total sales for Coffee Heaven is $8367.77"

# # test using question and generation from above
# answer_grader_prompt_formatted = answer_grader_prompt.format(
#   question=question, generation=answer
# )

# result = llm_json_mode.invoke(
#   [SystemMessage(content=answer_grader_instructions)]
#   + [HumanMessage(content=answer_grader_prompt_formatted)]
# )

# json.loads(result.content)