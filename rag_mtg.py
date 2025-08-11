import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain.chat_models import init_chat_model
import getpass
import os


if not os.environ.get("ANTHROPIC_API_KEY"):
  os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

llm = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")

embeddings = DeterministicFakeEmbedding(size=4096)

vector_store = InMemoryVectorStore(embeddings)

file_path = "./mtg_rules.txt" 

with open(file_path, 'r') as file:
    # Read the entire content of the file
    content = file.read()
    print("File content:")
    print(content)
    
    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
all_splits = text_splitter.split_text(content)

print(f"Split blog post into {len(all_splits)} sub-documents.")

# Index chunks
_ = vector_store.add_texts(texts=all_splits)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])

graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

print("RETRIEVED")
response = graph.invoke({"question": "What are the rules of Daybound?"})
print(response["answer"])