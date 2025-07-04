import os
import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

openai_api_key = os.getenv("OPENAI_API_KEY")

bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(["https://habr.com/ru/articles/871226/"])

docs = loader.load()
# print(f"docs {docs}")
print(f"docs len {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
# all_splits = text_splitter.split_text(str(docs))
all_splits = text_splitter.split_documents(docs)
# print(f"all_splits {all_splits}")
print(f"splits count {len(all_splits)}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
print(embeddings)
vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# ids = vector_store.add_texts(all_splits)
ids = vector_store.add_documents(all_splits)
print("data saved ./chroma_db")