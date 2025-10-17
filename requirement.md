# SYSTEM INSTRUCTION (for Cursor)
Use this document as the **single source of truth** for all project design decisions.
Modify the entire codebase accordingly. Remove any template code that is irrelevant.
Ensure the system runs correctly with:
- docker compose up
- API endpoints functional
- tests passing


# Ontology-Aware Memory System for an LLM Agent 

## Goal

Build a small but production‑minded service that gives an LLM a **memory that automatically grows from session to session**, grounded in data from an **external Postgres database** representing basic business processes. The service must:

1. **Persist and evolve memory** across user sessions (short‑term → long‑term).
2. **Reference & understand business processes** by joining to an existing Postgres schema (e.g., customers, orders, invoices, tasks).
3. **Retrieve, summarize, and inject** the most relevant memories and database facts into prompts.
4. Expose a minimal **HTTP API** for chat and memory inspection.

---

## Provided Business Context & Schema (External DB)

**Concept**: A lightweight ERP slice that tracks sales orders → work orders → invoices → payments, plus support tasks.xf

### Tables (you will **create & seed** these in Postgres)

Use SQL below as a starting point; adjust as needed. Include migrations.

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Domain data (the "external" schema)
CREATE SCHEMA IF NOT EXISTS domain;

CREATE TABLE IF NOT EXISTS domain.customers (
  customer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  industry TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS domain.sales_orders (
  so_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id UUID NOT NULL REFERENCES domain.customers(customer_id),
  so_number TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('draft','approved','in_fulfillment','fulfilled','cancelled')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS domain.work_orders (
  wo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  so_id UUID NOT NULL REFERENCES domain.sales_orders(so_id),
  description TEXT,
  status TEXT NOT NULL CHECK (status IN ('queued','in_progress','blocked','done')),
  technician TEXT,
  scheduled_for DATE
);

CREATE TABLE IF NOT EXISTS domain.invoices (
  invoice_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  so_id UUID NOT NULL REFERENCES domain.sales_orders(so_id),
  invoice_number TEXT UNIQUE NOT NULL,
  amount NUMERIC(12,2) NOT NULL,
  due_date DATE NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('open','paid','void')),
  issued_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS domain.payments (
  payment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  invoice_id UUID NOT NULL REFERENCES domain.invoices(invoice_id),
  amount NUMERIC(12,2) NOT NULL,
  method TEXT,
  paid_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS domain.tasks (
  task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id UUID REFERENCES domain.customers(customer_id),
  title TEXT NOT NULL,
  body TEXT,
  status TEXT NOT NULL CHECK (status IN ('todo','doing','done')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

```

### Minimal Seed Ideas (examples)

- Customers: “Gai Media” (Entertainment), “PC Boiler” (Industrial).
- Sales Orders: `SO-1001` (Gai Media, “Album Fulfillment”), `SO-2002` (PC Boiler, “On‑site repair”).
- Work Orders: for each SO (e.g., “Pick‑pack albums”, “Replace valve”).
- Invoices: amounts/due dates; mark one paid via Payments.
- Tasks: support items (“Investigate shipping SLA for Gai Media”).

> Deliver seeds as SQL and/or a script executable via docker-compose up.
> 

---

## Scenario‑Based User Journeys (Examples)

### 1) Overdue invoice follow‑up with preference recall

**Context**: Finance agent wants to nudge customer while honoring delivery preferences.
**Prior memory/DB**: `domain.invoices` shows Kai Media has `INV-1009` due 2025‑09‑30; memory: "prefers Friday deliveries".
**User**: "Draft an email for Kai Media about their unpaid invoice and mention their preferred delivery day for the next shipment."
**Expected**: Retrieval surfaces `INV-1009` (open, amount, due date) + memory preference. Reply mentions invoice details and references Friday delivery. **Memory update**: add episodic note that an invoice reminder was initiated today.

### 2) Reschedule a work order based on technician availability

**Context**: Ops manager wants to move a work order.
**Prior memory/DB**: `domain.work_orders` for SO‑1001 is `queued`, `technician=Alex`, `scheduled_for=2025‑09‑22`.
**User**: "Reschedule Kai Media’s pick‑pack WO to Friday and keep Alex assigned."
**Expected**: Tool queries the WO row, updates plan (you may simulate update by returning SQL suggestion), and stores a semantic memory: "Kai Media prefers Friday; align WO scheduling accordingly." **Trace** lists WO row used.

### 3) Ambiguous entity disambiguation (two customers with similar names)

**Context**: Two customers: "Kai Media" and "Kai Media Europe".
**User**: "What’s Kai’s latest invoice?"
**Expected**: System asks a **single step clarification** only if top‑k semantic & deterministic match scores are within a small margin; otherwise chooses the higher‑confidence entity. **Memory update**: store user’s clarification for future disambiguation (alias mapping).

### 4) NET terms learning from conversation

**Context**: Terms aren’t in DB; user states them.
**User**: "Remember: TC Boiler is NET15 and prefers ACH over credit card."
**Expected**: Create semantic memory entries (terms, payment method). On later: "When should we expect payment from TC Boiler on INV‑2201?" → system infers due date using invoice `issued_at` + NET15 from memory.

### 5) Partial payments and balance calculation

**Context**: An invoice has two `domain.payments` rows totaling less than invoice amount.
**User**: "How much does TC Boiler still owe on INV‑2201?"
**Expected**: Join payments to compute remaining balance; store episodic memory that user asked about balances for this customer (improves future weighting for finance intents).

### 6) SLA breach detection from tasks + orders

**Context**: Support tasks reference an SLA window.
**DB**: `domain.tasks` has "Investigate shipping SLA for Kai Media" (status=todo, created 10 days ago).
**User**: "Are we at risk of an SLA breach for Kai Media?"
**Expected**: Retrieve open task age + SO status; reply flags risk and suggests next steps; memory logs a risk tag to increase importance for related recalls.

### 7) Conflicting memories → consolidation rules

**Context**: Two sessions recorded delivery preference as Thursday vs Friday.
**User**: "What day should we deliver to Kai Media?"
**Expected**: Consolidation picks the most recent or most reinforced value; reply cites confidence and offers to confirm. If confirmed, demote conflicting memory via decay and add a corrective semantic memory.

### 8) Multilingual/alias handling

**Context**: User mixes languages.
**User**: "Recuérdame que Kai Media prefiere entregas los viernes."
**Expected**: NER detects "Kai Media"; memory stored in English canonical form with original text preserved; future English queries retrieve it. Add alias mapping for multilingual mentions.

### 9) Cold‑start grounding to DB facts

**Context**: No prior memories.
**User**: "What’s the status of TC Boiler’s order?"
**Expected**: System returns purely **DB‑grounded** answer from `domain.sales_orders`/`work_orders`, and creates an episodic memory summarizing the question and the returned facts.

### 10) Active recall to validate stale facts

**Context**: Preference older than 90 days with low reinforcement.
**User**: "Schedule a delivery for Kai Media next week."
**Expected**: Before finalizing, system asks: "We have Friday preference on record from 2025‑05‑10; still accurate?" If confirmed, resets decay; if changed, updates semantic memory and writes a summary.

### 11) Cross‑object reasoning (SO → WO → Invoice chain)

**User**: "Can we invoice as soon as the repair is done?"
**Expected**: Use chain: if `work_orders.status=done` for SO, and no `invoices` exist, recommend generating an invoice; if invoice exists and is open, suggest sending it. Memory stores the policy preference "Invoice immediately upon WO=done".

### 12) Conversation‑driven entity linking with fuzzy match

**Context**: User types "Kay Media".
**User**: "Open a WO for Kay Media for packaging."
**Expected**: Fuzzy match exceeds threshold to "Kai Media"; system confirms once, then stores alias in `app.entities` to avoid repeated confirmations.

### 13) Policy & PII guardrail memory

**User**: "Remember my personal cell: 415‑555‑0199 for urgent alerts."
**Expected**: Redact before storage per PII policy (store masked token + purpose). On later: "Alert me about any overdue invoice today" → system uses masked contact, not raw PII.

### 14) Session window consolidation example

**Flow**: Three sessions discuss TC Boiler terms, a rush WO, and a follow‑up payment plan.
**Expected**: `/consolidate` generates a single summary capturing NET15, rush WO linked to SO‑2002, and payment plan; subsequent retrieval uses this summary to answer "What are TC Boiler’s agreed terms and current commitments?"

### 15) Audit trail / explainability

**User**: "Why did you say Kai Media prefers Fridays?"
**Expected**: `/explain` returns memory IDs, similarity scores, and the specific chat event that created the memory, plus any reinforcing confirmations.

### 16) Reminder creation from conversational intent

**User**: "If an invoice is still open 3 days before due, remind me."
**Expected**: Store semantic policy memory; on future `/chat` calls that involve invoices, system checks due dates and surfaces proactive notices.

### 17) Error handling when DB and memory disagree

**Context**: Memory says "SO‑1001 is fulfilled" but DB shows `in_fulfillment`.
**User**: "Is SO‑1001 complete?"
**Expected**: Prefer authoritative DB; respond with DB truth and mark the outdated memory for decay and corrective summary.

### 18) Task completion via conversation

**User**: "Mark the SLA investigation task for Kai Media as done and summarize what we learned."
**Expected**: Return SQL patch suggestion (or mocked effect), store the summary as semantic memory for future reasoning.

---

## Memory Model (Your Service’s Schema)

Create a separate **app** schema for memory.

```sql
CREATE SCHEMA IF NOT EXISTS app;

-- Raw message events
CREATE TABLE IF NOT EXISTS app.chat_events (
  event_id BIGSERIAL PRIMARY KEY,
  session_id UUID NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Extracted entities (from messages + external DB linking)
CREATE TABLE IF NOT EXISTS app.entities (
  entity_id BIGSERIAL PRIMARY KEY,
  session_id UUID NOT NULL,
  name TEXT NOT NULL,
  type TEXT NOT NULL, -- e.g., customer, order, invoice, person, topic
  source TEXT NOT NULL, -- 'message' | 'db'
  external_ref JSONB,   -- e.g., {"table":"domain.customers","id":"..."}
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Memory chunks (vectorized)
CREATE TABLE IF NOT EXISTS app.memories (
  memory_id BIGSERIAL PRIMARY KEY,
  session_id UUID NOT NULL,
  kind TEXT NOT NULL, -- 'episodic','semantic','profile','commitment','todo'
  text TEXT NOT NULL,
  embedding vector(1536),
  importance REAL NOT NULL DEFAULT 0.5, -- 0..1 subjective weight
  ttl_days INT, -- optional expiry for short‑term
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Cross‑session consolidation log
CREATE TABLE IF NOT EXISTS app.memory_summaries (
  summary_id BIGSERIAL PRIMARY KEY,
  user_id TEXT NOT NULL, -- logical user handle
  session_window INT NOT NULL, -- N‑session window used
  summary TEXT NOT NULL,
  embedding vector(1536),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memories_embedding ON app.memories USING ivfflat (embedding vector_cosine);
CREATE INDEX IF NOT EXISTS idx_memory_summaries_embedding ON app.memory_summaries USING ivfflat (embedding vector_cosine);

```

Suggestions

- **Episodic memory**: store salient events per session.
- **Semantic memory**: distilled facts (e.g., “Gai Media prefers NET30”).
- **Consolidation**: after each session, write/update a rolling `memory_summaries` for the user (e.g., last N=3 sessions).
- **Linking**: detect entities in text and link to domain rows when possible (by exact/semantic match on names/IDs).
- **Retrieval**: for a new message, fetch top‑k from `app.memories` and `app.memory_summaries` and **augment** with live **domain** facts (SQL joins by entity linkage).
- **For cost safety**: make embedding & LLM providers pluggable.

---

## Functional Requirements

1. **/chat (POST)**
    
    Request: `{ user_id, session_id?, message }`
    
    Behavior:
    
    - Create a session if not provided.
    - Ingest `message` into `app.chat_events`.
    - Extract entities (NER + rule/SQL lookups into `domain.*`).
    - Generate/update memories (`app.memories`) with embeddings.
    - Retrieve top‑k memories + relevant `domain.*` facts.
    - Return an assistant reply that **references** domain facts (e.g., order status, invoice due).
    Response: `{ reply, used_memories: [...], used_domain_facts: [...] }`.
2. **/memory (GET)**
    
    Query: `{ user_id, k? }`
    
    Returns top memories & summaries for inspection.
    
3. **/consolidate (POST)**
    
    Request: `{ user_id }`
    
    Consolidate last N sessions → write/update `app.memory_summaries`.
    
4. **/entities (GET)**
    
    Query: `{ session_id }` → list detected entities and external refs.
    

> Bonus: /explain (GET) to show which SQL rows and memory chunks influenced the reply (traceability).
> 

---

## Non‑Functional Requirements

- **Idempotency**: re‑sending the same chat event should not duplicate memories (e.g., hash content).
- **PII Safety**: configurable redaction of emails/phones before storage.
- **Observability**: log retrieval candidates with scores & latencies.
- **Performance**: p95 /chat under 800ms for retrieval & DB joins with the provided seed size.
- **Security**: do not log secrets; support `.env` with example.

---

## Implementation Constraints

- Language: **Python**.
- DB: **Postgres 15+ with pgvector**. No external vector DB.
- Embeddings: OpenAI. Abstract behind an interface.
- Containerization: **docker‑compose** with services: `api`, `db`, `migrations`, `seed`.
- Migrations: Prisma/Knex/SQLAlchemy/Alembic/flyway—your choice.
- Tests: basic unit + an E2E happy path.

---

## Retrieval & Prompting Contract

- Use **hybrid retrieval**:
    1. vector search over `app.memories` and `app.memory_summaries`,
    2. keyword/SQL filters (e.g., by entity_type or session scope), then
    3. **augment** with **domain facts** via targeted queries (e.g., latest invoice for a linked customer).
- Construct a **system prompt** section listing canonical facts gathered from the DB (with IDs), followed by memory snippets.

**Example Augmentation (pseudo)**

```
System: Use the following facts and memories.
DB Facts:
- Customer[Gai Media] industry=Entertainment; open_invoices=1; last_invoice: INV-1009 (due 2025‑09‑30, $1,200, status=open)
- SO[SO-1001] status=in_fulfillment; WO status=queued (tech: Alex)
Memories:
- [semantic 0.78] Gai Media prefers NET30 and Friday deliveries.
- [episodic 0.66] User asked to expedite Gai Media order on 2025‑09‑12.

```

---

## Acceptance Criteria (Executable Checks)

Provide a **`scripts/acceptance.sh`** (or `.py`) that runs after `docker-compose up`:

1. **Seed check**: counts of `domain.*` tables > 0.
2. **Chat**: POST `/chat` with a prompt like “What’s the status of Gai Media’s order and any unpaid invoices?” → response must mention the correct SO/WO/Invoice.
3. **Memory growth**:
    - Session A: user says “They prefer Friday deliveries.”
    - Session B (new): ask “When should we deliver for Gai Media?” → reply includes the preference via memory retrieval.
4. **Consolidation**: POST `/consolidate` → `app.memory_summaries` row created/updated.
5. **Entities**: GET `/entities?session_id=...` returns links to `domain.customers`.

Return **non‑zero exit** if any check fails.

---

---

## What to Submit

1. **Repo link** (GitHub/GitLab) with:
    - `docker-compose.yml` (db, api, migration, seed).
    - Migrations & seeds.
    - Source code with clear module boundaries (ingest, link, embed, store, retrieve, augment, respond).
    - **README** with setup, `.env.example`, architecture diagram, and trade‑offs.
    - Tests + `scripts/acceptance.sh`.
2. **Short write‑up (≤1 page)** covering:
    - Memory lifecycle: extraction → storage → consolidation → retrieval.
    - Linking strategy (exact vs fuzzy; how you resolved collisions).
    - Prompt format and guardrails (PII, hallucination mitigation).
    - Future improvements.

---

## Evaluation Rubric (100 pts)

- **Correctness & AC pass** (25)
- **Memory growth & consolidation quality** (15)
- **Retrieval quality (DB + memory)** (15)
- **Architecture & code quality** (20)
- **Data modeling & migrations** (10)
- **Observability & tests** (5)
- **Performance & efficiency** (5)
- **Security & configs** (5)

---

## Hints & Guardrails

- Prefer **deterministic linking** first (IDs, exact names) then fallback **semantic match** with thresholds.
- Store **why** a memory was created (heuristic score, rule matched).
- Keep vector dimensions configurable; default 1536.
- Cache domain facts per request to minimize N+1 queries.
- Avoid over‑storing; summarize long chats.
- Make it easy to swap embedding/LLM providers.

---

## Sample Requests (for your README)

```bash
# 1) User asks about a customer’s status
curl -sX POST <http://localhost:8080/chat> \\
 -H 'Content-Type: application/json' \\
 -d '{
  "user_id":"demo-user",
  "message":"What is the status of Gai Media\\'s order and any unpaid invoices?"
 }'

# 2) Add a preference into memory (Session A)
curl -sX POST <http://localhost:8080/chat> \\
 -H 'Content-Type: application/json' \\
 -d '{
  "user_id":"demo-user",
  "session_id":"00000000-0000-0000-0000-000000000001",
  "message":"Remember: Gai Media prefers Friday deliveries."
 }'

# 3) New session, recall preference (Session B)
curl -sX POST <http://localhost:8080/chat> \\
 -H 'Content-Type: application/json' \\
 -d '{
  "user_id":"demo-user",
  "message":"When should we deliver for Gai Media?"
 }'

```

---

**Good luck!**