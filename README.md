# Walt Bot 🤖✍️

> An AI-powered writing assistant that combines the wisdom of Walt Whitman, the mysticism of Lon Milo DuQuette, and the wonder of Carl Sagan.

Walt Bot is a sophisticated FastAPI application featuring agentic RAG (Retrieval-Augmented Generation) with LangGraph, vector similarity search, and Named Entity Recognition to help writers organize, search, and draw insights from their creative work.

## ✨ Features

### 🧠 Intelligent Multi-Modal RAG System
- **Agentic Routing**: Automatically routes queries to the appropriate knowledge base (passage archives, freewriting, or general chat)
- **Context-Aware Responses**: Retrieves relevant context from vector stores before generating responses
- **Conversational Memory**: Maintains conversation history for coherent multi-turn dialogues

### 📚 Vector-Powered Knowledge Management
- **Dual Collection System**: Separate vector stores for structured passages and freewriting
- **Semantic Search**: ChromaDB-powered similarity search with Ollama embeddings
- **Smart Text Chunking**: Automatic text splitting with configurable overlap for optimal retrieval
- **Raw Text Ingestion**: Direct upload of freewriting with automatic chunking and hashing

### 🔍 Named Entity Recognition (NER)
- **Entity Extraction**: Identifies people, organizations, locations, and dates using BERT-based NER
- **NER-Powered Search**: Query your writing based on extracted entities
- **Entity Aggregation**: Consolidated view of all entities found in retrieved passages

### 📝 Journal & Passage Management
- **Hierarchical Organization**: Journals contain multiple passages
- **CRUD Operations**: Full create, read, update, delete functionality
- **Timestamp Tracking**: Automatic creation and update timestamps
- **In-Memory Database**: Fast, lightweight storage (easily adaptable to PostgreSQL/SQLite)

### 🤖 LangGraph Workflows
- **Visual State Machines**: Graph-based workflows for complex AI operations
- **Conditional Routing**: Dynamic path selection based on query content
- **State Persistence**: MemorySaver checkpointing for stateful conversations
- **Modular Node Architecture**: Reusable, composable processing nodes

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
├─────────────────────────────────────────────────────────────┤
│  Routers                                                     │
│  ├── /journals          - Journal CRUD operations           │
│  ├── /passages          - Passage CRUD operations           │
│  ├── /vector-ops        - Vector DB operations + NER        │
│  └── /langgraph         - Agentic chat endpoint             │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌──────────────┐    ┌──────────────────┐
│  LangGraph    │    │  Vector DB   │    │  NER Pipeline    │
│  Services     │    │  Service     │    │  (Transformers)  │
├───────────────┤    ├──────────────┤    ├──────────────────┤
│ • Main graph  │───▶│ • ChromaDB   │    │ • BERT NER Model │
│ • NER graph   │    │ • Embeddings │    │ • Entity Extract │
│ • Text graph  │    │ • Collections│    │ • Aggregation    │
└───────────────┘    └──────────────┘    └──────────────────┘
        │                     │
        └──────────┬──────────┘
                   ▼
          ┌─────────────────┐
          │  Ollama (Local) │
          ├─────────────────┤
          │ • Mistral LLM   │
          │ • Nomic Embed   │
          └─────────────────┘
```

### LangGraph Agentic Flow

```
User Query
    │
    ▼
┌─────────────┐
│ Route Node  │──┬──[passages]───▶ Extract Passages ──┐
└─────────────┘  │                                     │
                 ├──[freewriting]─▶ Extract Text ──────┤
                 │                                     │
                 └──[chat]────────▶ General Chat      │
                                                       │
                                                       ▼
                                            ┌───────────────────┐
                                            │ Answer Generation │
                                            │  with Context     │
                                            └───────────────────┘
                                                       │
                                                       ▼
                                                  Response
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (3.13 recommended)
- [Ollama](https://ollama.ai/) installed and running locally
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/walt-bot.git
   cd walt-bot
   ```

2. **Install Ollama models**
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**
   ```bash
   uvicorn app.main:app --reload --port 8080
   ```

5. **Access the API**
   - API: http://127.0.0.1:8080
   - Interactive Docs: http://127.0.0.1:8080/docs
   - Alternative Docs: http://127.0.0.1:8080/redoc

## 📖 API Documentation

### LangGraph Chat (Agentic RAG)

**POST** `/langgraph/chat`

Intelligent routing-based chat with automatic context retrieval.

```json
{
  "input": "What did I write about in my dream journal?"
}
```

**Response:**
```json
{
  "route": "passages",
  "answer": "In your dream journal, you explored...",
  "sources": [...],
  "message_memory": [...]
}
```

### Vector Operations

#### Ingest Text (Freewriting)

**POST** `/vector-ops/ingest-text`

Content-Type: `multipart/form-data`

```bash
curl -X POST "http://127.0.0.1:8080/vector-ops/ingest-text" \
  -F "text=Your freewriting content here..."
```

#### Ingest Structured Passages

**POST** `/vector-ops/ingest-json`

```json
[
  {
    "id": "passage_001",
    "text": "The cosmos is within us...",
    "metadata": {
      "author": "Carl Sagan",
      "source": "Cosmos"
    }
  }
]
```

#### Search with RAG

**POST** `/vector-ops/search-text`

```json
{
  "query": "What did I write about the stars?",
  "k": 5
}
```

#### NER-Powered Search

**POST** `/vector-ops/ner-search-text`

Extract entities from relevant passages and answer based on them.

```json
{
  "query": "Who are the people I mentioned?",
  "k": 10
}
```

**Response:**
```json
{
  "answer": "Based on the entities found...",
  "entities": {
    "PERSON": ["Walt Whitman", "Carl Sagan"],
    "ORG": ["NASA"],
    "LOC": ["New York"],
    "DATE": ["2026"],
    "OTHER": []
  },
  "query": "Who are the people I mentioned?"
}
```

### Journal & Passage Management

#### Create Journal

**POST** `/journals/`

```json
{
  "id": 1,
  "title": "My Creative Writing",
  "created_at": "2026-01-15T10:30:00"
}
```

#### Create Passage

**POST** `/passages/journals/{journal_id}/new_passage`

```json
{
  "id": 1,
  "journal_id": 1,
  "title": "Midnight Thoughts",
  "content": "The stars whispered secrets...",
  "created_at": "2026-01-15T23:45:00"
}
```

#### Get All Passages from Journal

**GET** `/passages/journals/{journal_id}/passages/`

## 🛠️ Configuration

### Environment Variables

Create a `.env` file (optional):

```env
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIR=app/chroma_store
```

### LLM Settings

Edit `langgraph_service.py` or `vector_langgraph_service.py`:

```python
llm = ChatOllama(
    model="mistral",      # Change model here
    temperature=0.2       # Adjust creativity (0.0 - 1.0)
)
```

### Vector Store Collections

Two collections are used:
- `passage_archive` - Structured passages from journals
- `freewriting` - Raw text chunks from freewriting uploads

## 📁 Project Structure

```
walt-bot/
├── app/
│   ├── main.py                          # FastAPI application
│   ├── models/
│   │   ├── journal_model.py             # Journal Pydantic model
│   │   └── passage_model.py             # Passage Pydantic model
│   ├── routers/
│   │   ├── journals.py                  # Journal CRUD endpoints
│   │   ├── passages.py                  # Passage CRUD endpoints
│   │   ├── langgraph_ops.py             # Agentic chat endpoint
│   │   └── vector_ops.py                # Vector DB + NER endpoints
│   ├── services/
│   │   ├── langgraph_service.py         # Main agentic graph
│   │   ├── vector_langgraph_service.py  # Vector-specific graphs
│   │   └── vectordb_service.py          # ChromaDB + NER operations
│   └── chroma_store/                    # Vector DB persistence
├── requirements.txt
└── README.md
```

## 🎯 Use Cases

### 1. **Creative Writing Assistant**
Upload freewriting, then ask Walt Bot to help you identify themes, characters, or plot points.

```bash
# Upload your writing
curl -X POST "http://localhost:8080/vector-ops/ingest-text" \
  -F "text@my_novel_draft.txt"

# Ask for insights
curl -X POST "http://localhost:8080/langgraph/chat" \
  -H "Content-Type: application/json" \
  -d '{"input": "What are the recurring themes in my writing?"}'
```

### 2. **Research Note Organization**
Store research notes as passages and use semantic search to find connections.

```python
import requests

# Ingest research notes
passages = [
    {"id": "note_001", "text": "Quantum entanglement allows...", "metadata": {"topic": "physics"}},
    {"id": "note_002", "text": "Poetic meter in Walt Whitman...", "metadata": {"topic": "poetry"}}
]

requests.post("http://localhost:8080/vector-ops/ingest-json", json=passages)

# Search across topics
response = requests.post(
    "http://localhost:8080/vector-ops/search-text",
    json={"query": "connections between physics and poetry", "k": 5}
)
```

### 3. **Entity Tracking Across Documents**
Find all mentions of specific people, places, or organizations.

```python
response = requests.post(
    "http://localhost:8080/vector-ops/ner-search-text",
    json={"query": "Who are the scientists I've written about?", "k": 20}
)

print(response.json()["entities"]["PERSON"])
```

## 🧪 Development



### Code Structure Guidelines

- **Routers**: Handle HTTP requests/responses, minimal business logic
- **Services**: Core business logic, LangGraph workflows, DB operations
- **Models**: Pydantic models for data validation

### Adding a New LangGraph Node

1. Define node function in `langgraph_service.py`:
   ```python
   def my_new_node(state: GraphState) -> GraphState:
       # Process state
       return {"new_field": result}
   ```

2. Add to graph builder:
   ```python
   build.add_node("my_node", my_new_node)
   build.add_edge("previous_node", "my_node")
   ```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain & LangGraph** - For the amazing agentic AI framework
- **Ollama** - For making local LLMs accessible
- **ChromaDB** - For the vector database
- **Hugging Face** - For the transformers and NER models
- **Walt Whitman, Carl Sagan, Lon Milo DuQuette** - For inspiring the personality of Walt Bot

## 🔮 Possible Roadmap

- [ ] Add unit tests, integration tests, and smoke tests
- [ ] Add support for document upload (PDF, DOCX)
- [ ] Implement user authentication and multi-user support
- [ ] Add conversation export functionality
- [ ] Support for additional LLM providers (Anthropic, OpenAI)
- [ ] Web UI for easier interaction
- [ ] PostgreSQL/SQLite backend option
- [ ] Batch processing for large document collections
- [ ] Fine-tuned embeddings for creative writing
- [ ] Graph visualization of entity relationships

## 📧 Contact

Project Link: [https://github.com/sciencefixion/AH_walt_bot](https://github.com/sciencefixion/AH_walt_bot)

---

**Built with ❤️ by writers, for writers**
