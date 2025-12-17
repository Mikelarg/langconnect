# LangConnect
LangConnect — это сервис RAG (Retrieval-Augmented Generation), созданный с использованием FastAPI и LangChain. Он предоставляет REST API для управления коллекциями и документами, а также PostgreSQL и pgvector для хранения векторных данных.

## Установка

### Требования

- Docker и Docker Compose
- Python 3.11 или выше

### Запуск с Docker

1. Скачайте репозиторий:
   ```bash
   git clone git@github.com:Mikelarg/giga_agent_langconnect.git
   cd giga_agent_langconnect
   ```

2. Заполните ENV переменные в файле `.env`. Пример .env файла: [.env.example](.env.example)

3. Запустите сервис:
   ```bash
   docker-compose up -d
   ```

   Это:
   - Запустит PostgreSQL бд с pgvector расширением
   - Забилдит и запустит LangConnect API

4. Подключение к API:
   - API документация: http://localhost:8833/docs
   - Health check: http://localhost:8833/health

5. Подключите LangConnect к GigaAgent
   - С помощью ENV переменных `LANGCONNECT_API_URL` и `LANGCONNECT_API_SECRET_TOKEN`
   - Пример:
   ```
   LANGCONNECT_API_URL=http://host.docker.internal:8833
   LANGCONNECT_API_SECRET_TOKEN=123
   ```
## Endpoints

### Collections

#### `/collections` (GET)

List all collections.

#### `/collections` (POST)

Create a new collection.

#### `/collections/{collection_id}` (GET)

Get a specific collection by ID.

#### `/collections/{collection_id}` (DELETE)

Delete a specific collection by ID.

### Documents

#### `/collections/{collection_id}/documents` (GET)

List all documents in a specific collection.

#### `/collections/{collection_id}/documents` (POST)

Create a new document in a specific collection.

#### `/collections/{collection_id}/documents/{document_id}` (DELETE)

Delete a specific document by ID.

#### `/collections/{collection_id}/documents/search` (POST)

Search for documents using semantic search.
