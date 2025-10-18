# Project Proposal: Ontology-Aware Memory System for LLM Agent

## Solution Architecture

### Core Pipeline

```
1. User Input → ChatRequest
   ↓
2. API-level Clarification Response Detection → _handle_clarification_response()
   ├── Is Clarification Response → Process directly and return
   └── Not Clarification Response → Continue Pipeline
   ↓
3. Quick Intent Detection → Determine if full processing is needed
   ├── General Conversation → Simplified flow (Short-term Memory)
   └── Business-related → Full flow
   ↓
4. PII Detection → Detect and mask sensitive information
   ↓
5. Entity Extraction → EntityService.extract_entities()
   ↓
6. Disambiguation Service Integration → DisambiguationService.decide_disambiguation()
   ├── Needs Clarification → Return clarification question, store ChatEvent
   └── No Clarification Needed → Continue normal flow
   ↓
7. Generate Embedding → EmbeddingService.generate_embedding()
   ↓
8. Retrieve Context → RetrievalService.retrieve_context()
   ↓
9. Build Prompt → PromptContext
   ↓
10. Generate LLM Response → LLMService.generate_response()
   ↓
11. Memory Processing → IntentBasedMemoryExtractor
    ├── ACTION Intent → Episodic Memory
    ├── PREFERENCE Intent → Semantic Memory
    └── Other Intents → Short-term Memory or Skip
   ↓
12. Memory Storage → MemoryService.create_memory()
   ↓
13. Store Chat Event → ChatEvent
   ↓
14. Return Response → ChatResponse
```

## Technical Implementation

### Technology Stack

- **Backend**: FastAPI, SQLModel, PostgreSQL with pgvector
- **AI Services**: OpenAI API for embeddings and LLM interactions
- **Database**: PostgreSQL 15+ with pgvector extension for vector similarity search
- **Deployment**: Docker Compose for containerized deployment
- **Migration**: Alembic for database schema management

### Code Structure

```
backend/
├── app/
│   ├── api/routes/          # API endpoints
│   │   ├── chat.py         # Main chat endpoint
│   │   ├── memory.py       # Memory retrieval
│   │   ├── consolidate.py # Memory consolidation
│   │   ├── entities.py     # Entity management
│   │   └── explain.py      # Explainability
│   ├── core/               # Core configuration
│   ├── models/             # Data models
│   │   ├── domain.py       # Business domain models
│   │   ├── memory.py       # Memory system models
│   │   └── chat.py         # Chat models
│   ├── services/           # Business services
│   │   ├── hybrid_chat_pipeline.py    # Main pipeline
│   │   ├── entity_service.py         # Entity extraction
│   │   ├── memory_service.py         # Memory management
│   │   ├── retrieval_service.py     # Context retrieval
│   │   ├── llm_service.py           # LLM interactions
│   │   ├── disambiguation_service.py # Entity disambiguation
│   │   └── intent_based_memory_extractor.py # Intent analysis
│   └── alembic/            # Database migrations
└── scripts/                # Utility scripts
```

### API Endpoints

- **POST /api/v1/chat/** - Main chat interface with memory integration
- **GET /api/v1/memory/** - Retrieve user memories with filtering
- **POST /api/v1/consolidate/** - Consolidate memories across sessions
- **GET /api/v1/entities/** - List detected entities for a session
- **GET /api/v1/explain/** - Explain memory retrieval decisions - Not implemented


# Database Architecture Diagram

## Database Schema Structure

### Domain Schema (Business Entity Layer)
```
Customer Table
├── SalesOrder Table
│   ├── WorkOrder Table
│   └── Invoice Table
│       └── Payment Table
└── Task Table
```

### App Schema (Application Layer)
```
ChatEvent Table
Entity Table
Memory Table (with vector embeddings)
MemorySummary Table (cross-session summaries)
```

## Pipeline and Database Interaction Flow

```
...
RetrievalService → Memory Table (retrieve memories)
                → MemorySummary Table (retrieve summaries)
                → Domain Tables (query business data)
    ↓
LLMService (generate response)
    ↓
MemoryService → Memory Table (store new memories)
    ↓
...
```


## Getting Started
```bash
# 1. Clone the project
git clone https://github.com/LUDingfu/Ontology-Aware-Memory-System-for-LLM-Agent.git
cd ontology-aware-memory-system

# 2. Configure environment variables
cp .env.example .env
# Edit .env file and set OpenAI API Key

# 3. Start services
docker-compose up -d

# 4. Run tests
./scripts/acceptance.sh
```
