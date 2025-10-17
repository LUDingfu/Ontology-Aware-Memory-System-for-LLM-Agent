# Ontology-Aware Memory System for LLM Agent - 项目规范

## 项目概述

基于FastAPI框架构建一个生产级的LLM智能体记忆系统，该系统能够：
1. 跨会话持久化和演进记忆（短期→长期）
2. 引用和理解外部PostgreSQL数据库中的业务流程
3. 检索、总结和注入最相关的记忆和数据库事实到提示中
4. 提供最小化的HTTP API用于聊天和记忆检查

## 技术架构

### 核心技术栈
- **后端框架**: FastAPI (Python)
- **数据库**: PostgreSQL 15+ with pgvector
- **ORM**: SQLModel (基于Pydantic和SQLAlchemy)
- **向量搜索**: pgvector扩展
- **嵌入模型**: OpenAI API (可配置)
- **容器化**: Docker Compose
- **数据库迁移**: Alembic

### 项目结构规划

```
backend/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py          # /chat 端点
│   │   │   ├── memory.py        # /memory 端点
│   │   │   ├── consolidate.py   # /consolidate 端点
│   │   │   ├── entities.py      # /entities 端点
│   │   │   └── explain.py       # /explain 端点 (bonus)
│   │   ├── deps.py              # 依赖注入
│   │   └── main.py              # API路由器
│   ├── core/
│   │   ├── config.py            # 配置管理
│   │   ├── db.py                # 数据库连接
│   │   └── security.py          # 安全相关
│   ├── models/
│   │   ├── domain.py            # 业务域模型 (customers, orders, etc.)
│   │   ├── memory.py            # 记忆系统模型
│   │   └── chat.py              # 聊天相关模型
│   ├── services/
│   │   ├── memory_service.py    # 记忆管理服务
│   │   ├── entity_service.py    # 实体识别和链接服务
│   │   ├── embedding_service.py # 嵌入生成服务
│   │   ├── retrieval_service.py # 检索服务
│   │   └── llm_service.py       # LLM交互服务
│   ├── utils/
│   │   ├── nlp.py               # NLP工具函数
│   │   ├── pii_detection.py     # PII检测和脱敏
│   │   └── consolidation.py    # 记忆整合工具
│   ├── alembic/                 # 数据库迁移
│   │   └── versions/
│   │       ├── 001_create_domain_schema.py
│   │       ├── 002_create_memory_schema.py
│   │       └── 003_seed_data.py
│   └── main.py                  # 应用入口
├── tests/
│   ├── test_chat.py
│   ├── test_memory.py
│   ├── test_entities.py
│   └── test_acceptance.py
├── scripts/
│   ├── acceptance.sh            # 验收测试脚本
│   └── seed_data.py             # 数据种子脚本
└── pyproject.toml
```

## 数据库设计

### Domain Schema (业务域)
```sql
-- 启用pgvector扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 业务域数据
CREATE SCHEMA IF NOT EXISTS domain;

-- 客户表
CREATE TABLE domain.customers (
  customer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  industry TEXT,
  notes TEXT
);

-- 销售订单表
CREATE TABLE domain.sales_orders (
  so_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id UUID NOT NULL REFERENCES domain.customers(customer_id),
  so_number TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('draft','approved','in_fulfillment','fulfilled','cancelled')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 工作订单表
CREATE TABLE domain.work_orders (
  wo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  so_id UUID NOT NULL REFERENCES domain.sales_orders(so_id),
  description TEXT,
  status TEXT NOT NULL CHECK (status IN ('queued','in_progress','blocked','done')),
  technician TEXT,
  scheduled_for DATE
);

-- 发票表
CREATE TABLE domain.invoices (
  invoice_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  so_id UUID NOT NULL REFERENCES domain.sales_orders(so_id),
  invoice_number TEXT UNIQUE NOT NULL,
  amount NUMERIC(12,2) NOT NULL,
  due_date DATE NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('open','paid','void')),
  issued_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 付款表
CREATE TABLE domain.payments (
  payment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  invoice_id UUID NOT NULL REFERENCES domain.invoices(invoice_id),
  amount NUMERIC(12,2) NOT NULL,
  method TEXT,
  paid_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 任务表
CREATE TABLE domain.tasks (
  task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id UUID REFERENCES domain.customers(customer_id),
  title TEXT NOT NULL,
  body TEXT,
  status TEXT NOT NULL CHECK (status IN ('todo','doing','done')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### App Schema (记忆系统)
```sql
-- 应用记忆系统
CREATE SCHEMA IF NOT EXISTS app;

-- 原始消息事件
CREATE TABLE app.chat_events (
  event_id BIGSERIAL PRIMARY KEY,
  session_id UUID NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 提取的实体
CREATE TABLE app.entities (
  entity_id BIGSERIAL PRIMARY KEY,
  session_id UUID NOT NULL,
  name TEXT NOT NULL,
  type TEXT NOT NULL, -- 'customer', 'order', 'invoice', 'person', 'topic'
  source TEXT NOT NULL, -- 'message' | 'db'
  external_ref JSONB,   -- {"table":"domain.customers","id":"..."}
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 记忆块 (向量化)
CREATE TABLE app.memories (
  memory_id BIGSERIAL PRIMARY KEY,
  session_id UUID NOT NULL,
  kind TEXT NOT NULL, -- 'episodic','semantic','profile','commitment','todo'
  text TEXT NOT NULL,
  embedding vector(1536),
  importance REAL NOT NULL DEFAULT 0.5, -- 0..1 主观权重
  ttl_days INT, -- 短期记忆的可选过期时间
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 跨会话整合日志
CREATE TABLE app.memory_summaries (
  summary_id BIGSERIAL PRIMARY KEY,
  user_id TEXT NOT NULL, -- 逻辑用户句柄
  session_window INT NOT NULL, -- 使用的N会话窗口
  summary TEXT NOT NULL,
  embedding vector(1536),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 向量搜索索引
CREATE INDEX idx_memories_embedding ON app.memories USING ivfflat (embedding vector_cosine);
CREATE INDEX idx_memory_summaries_embedding ON app.memory_summaries USING ivfflat (embedding vector_cosine);
```

## API端点设计

### 1. POST /chat
**功能**: 处理用户聊天消息，生成记忆并返回回复
```json
{
  "user_id": "string",
  "session_id": "uuid?",
  "message": "string"
}
```

**响应**:
```json
{
  "reply": "string",
  "used_memories": [{"memory_id": 1, "text": "...", "similarity": 0.85}],
  "used_domain_facts": [{"table": "customers", "id": "...", "data": {...}}],
  "session_id": "uuid"
}
```

### 2. GET /memory
**功能**: 获取用户的记忆和摘要
```json
{
  "user_id": "string",
  "k": 10
}
```

### 3. POST /consolidate
**功能**: 整合最后N个会话的记忆
```json
{
  "user_id": "string"
}
```

### 4. GET /entities
**功能**: 获取会话中检测到的实体
```json
{
  "session_id": "uuid"
}
```

### 5. GET /explain (Bonus)
**功能**: 解释回复的来源和推理过程

## 核心服务模块

### 1. Memory Service
- **记忆生成**: 从聊天消息中提取和存储记忆
- **记忆检索**: 基于向量相似度和关键词过滤
- **记忆整合**: 跨会话的记忆摘要和去重

### 2. Entity Service
- **实体识别**: NER + 规则/SQL查找
- **实体链接**: 精确匹配 + 模糊匹配
- **实体解析**: 处理歧义和别名

### 3. Embedding Service
- **嵌入生成**: OpenAI API或本地模型
- **向量存储**: pgvector数据库存储
- **相似度计算**: 余弦相似度搜索

### 4. Retrieval Service
- **混合检索**: 向量搜索 + 关键词过滤
- **数据库增强**: 通过实体链接获取域事实
- **相关性排序**: 基于相似度和重要性的排序

### 5. LLM Service
- **提示构建**: 系统提示 + 记忆 + 域事实
- **回复生成**: LLM API调用
- **回复后处理**: PII检测、格式化等

## 实现步骤

### 阶段1: 基础架构 (2-3小时)
1. **清理现有代码** ✅
2. **创建新的数据库迁移**
3. **更新配置和依赖**
4. **创建基础模型**

### 阶段2: 核心服务 (3-4小时)
1. **实现Memory Service**
2. **实现Entity Service**
3. **实现Embedding Service**
4. **实现Retrieval Service**

### 阶段3: API端点 (2-3小时)
1. **实现/chat端点**
2. **实现/memory端点**
3. **实现/consolidate端点**
4. **实现/entities端点**

### 阶段4: 测试和优化 (1-2小时)
1. **编写单元测试**
2. **实现验收测试脚本**
3. **性能优化**
4. **文档完善**

## 验收标准

### 功能验收
1. **种子数据检查**: domain.*表有数据
2. **聊天功能**: 能正确回答关于客户订单和发票的问题
3. **记忆增长**: 跨会话记忆检索工作
4. **整合功能**: 记忆摘要生成
5. **实体识别**: 能识别和链接实体

### 性能要求
- p95 /chat响应时间 < 800ms
- 支持并发用户访问
- 内存使用合理

### 安全要求
- PII检测和脱敏
- 环境变量配置
- 不记录敏感信息

## 技术决策

### 1. 记忆类型
- **Episodic**: 会话中的具体事件
- **Semantic**: 提炼的事实和偏好
- **Profile**: 用户/客户画像
- **Commitment**: 承诺和待办事项

### 2. 实体链接策略
- **精确匹配**: ID、确切名称
- **模糊匹配**: 语义相似度阈值
- **歧义处理**: 单步澄清机制

### 3. 记忆整合
- **滚动窗口**: 最后N个会话
- **重要性权重**: 基于使用频率和确认
- **冲突解决**: 最新或最确认的值

### 4. 检索策略
- **混合检索**: 向量 + 关键词
- **分层增强**: 记忆 → 域事实 → 实时查询
- **相关性排序**: 相似度 + 重要性 + 时效性

## 未来改进

1. **多语言支持**: 跨语言实体识别和记忆
2. **主动提醒**: 基于记忆的智能提醒
3. **记忆衰减**: 时间衰减和重要性调整
4. **联邦学习**: 跨用户记忆模式学习
5. **可视化**: 记忆图谱和实体关系可视化
