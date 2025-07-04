# Анализ документов с использованием RAG

![Интерфейс](./assets/demo.png)

## Описание

Проект реализует систему **Retrieval-Augmented Generation (RAG)** для поиска ошибок и несоответствий в загруженных документов и получения ответов на вопросы по ним на естественном языке.  
Целью является изучение и практическое применение RAG-архитектуры, а также построение простого веб-интерфейса на FastAPI для интерактивной работы с системой.

Система может быть использована как для **(PDF, DOCX, TXT и html)**

---

## Особенности

- Поддержка форматов: `.pdf`, `.docx`, `.txt`, `.html`.
- Индексация и хранение векторов с помощью [Chroma](https://www.trychroma.com/)
- Векторизация с использованием модели `text-embedding-3-small` от OpenAI
- Генерация ответов с помощью OpenAI GPT
- Поддержка многодокументного поиска
- Автоматический выбор источника при отсутствии явного указания

---

## Диаграмма взаимодействия ИИ-агентов

```mermaid
graph TD
    user[User] -->|Ask (+optional source)| retriever[Retriever Agent]
    retriever --> router[Router Agent]
    router -->|Choose source| sourceSelector[Source Selector]
    sourceSelector --> contextBuilder[Context Builder]
    contextBuilder --> llm[LLM]
    llm --> answer[Answer to User]
```
---

## Быстрый старт
- pip install -r requirements.txt
- OPENAI_API_KEY=your_openai_key
- запустить в консоли fastapi dev main.py     