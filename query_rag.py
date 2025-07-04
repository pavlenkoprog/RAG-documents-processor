import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

prompt = ChatPromptTemplate.from_template(
    """Ты полезный  ассистент который может ответить на вопросы на тему RAG систем.
    Используй следующий текст чтобы ответить на вопросы. Если в тексте нет нужной информации сообщи об этом!
    Вопрос: {question}
    Контекст: {context} 
    Ответ:"""
)

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)


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


def get_rag_answer(question):
    retrieved_docs = vector_store.similarity_search(question, k=3)
    # candidates = vector_store.similarity_search(question, k=3, filter={"source": "habr"})
    reranked = rerank_with_openai(question, retrieved_docs)
    top_docs = reranked[:3]

    docs_content = "\n".join(doc.page_content for doc in top_docs)
    # for doc in retrieved_docs:
    #     print(f"CONTEXT: {doc.page_content}")

    message = prompt.invoke({"question": question, "context": docs_content})
    answer = llm.invoke(message)
    return answer.content


question = "Что такое Mean Reciprocal Rank (MRR) ?"
print(get_rag_answer(question))

question = "Почему небо голубое ?"
print(get_rag_answer(question))
