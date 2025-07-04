import os
import shutil
from typing import List, Optional
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
)

openai_api_key = os.getenv("OPENAI_API_KEY")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)


def save_file_to_uploads(file, filename: str) -> str:
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file, buffer)
    return path


def load_document(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    elif ext in [".txt", ".log"]:
        return TextLoader(path).load()
    elif ext in [".doc", ".docx"]:
        return UnstructuredWordDocumentLoader(path).load()
    elif ext in [".html", ".htm"]:
        return UnstructuredHTMLLoader(path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def process_and_store(path: str):
    docs = load_document(path)
    filename = os.path.basename(path)
    for d in docs:
        d.metadata["source"] = filename
    splits = text_splitter.split_documents(docs)
    vector_store.add_documents(splits)
    print(f"Saved to ChromaDB: {path}")


def list_uploaded_files() -> List[str]:
    return sorted(os.listdir(UPLOAD_DIR))


def get_all_sources() -> List[str]:
    results = vector_store.get(include=["metadatas"])
    sources = [meta.get("source") for meta in results["metadatas"] if "source" in meta]
    return sorted(set(sources))


llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
prompt = ChatPromptTemplate.from_template(
    """Ты полезный ассистент который может разбираться в документах и находить в них ошибки или подозрительные моменты.
    Используй следующий текст чтобы ответить на вопросы. Если в тексте нет нужной информации сообщи об этом!
    Вопрос: {question}
    Контекст: {context} 
    Ответ:"""
)


def rerank_with_openai(question: str, docs):
    chunks = [doc.page_content for doc in docs]
    joined_chunks = "\n".join([f"[{i}]: {chunk}" for i, chunk in enumerate(chunks)])

    rerank_prompt = f"""Ты — интеллектуальная модель, которая сортирует текстовые фрагменты по их релевантности к вопросу.
    Вопрос: {question}
    Фрагменты:{joined_chunks}
    Верни отсортированный список индексов от наиболее релевантного к наименее (пример: 2, 0, 1):"""

    response = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key).invoke(rerank_prompt)
    ordered_indices = [int(i.strip()) for i in response.content.split(",") if i.strip().isdigit()]
    return [docs[i] for i in ordered_indices]


def get_rag_answer(question: str, source: Optional[str] = None) -> str:
    if source:
        docs = vector_store.similarity_search(question, k=5, filter={"source": source})
    else:
        best_source = find_best_source(question)  # агент для выбора подходящего файла
        docs = vector_store.similarity_search(question, k=5, filter={"source": best_source})

    reranked = rerank_with_openai(question, docs)
    top_docs = reranked[:3]

    context = "\n".join(doc.page_content for doc in top_docs)
    message = prompt.invoke({"question": question, "context": context})
    answer = llm.invoke(message)
    return answer.content


def find_best_source(question: str, k: int = 3) -> str:
    results = vector_store.similarity_search_with_score(question, k=20)
    scored_by_source = {}

    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        if source not in scored_by_source:
            scored_by_source[source] = []
        scored_by_source[source].append(score)
        print("score источника ", source, score)

    # Средний score на источник
    avg_scores = {src: sum(scores)/len(scores) for src, scores in scored_by_source.items()}
    best_source = min(avg_scores, key=avg_scores.get)  # чем ниже score — тем ближе

    return best_source
