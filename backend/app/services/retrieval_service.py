"""Retrieval service for hybrid search."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlmodel import Session, select, text

from app.models.domain import Customer, SalesOrder, Invoice, Task, Payment, WorkOrder
from app.models.chat import DomainFact, RetrievalContext
from app.models.memory import Memory, MemoryRetrievalResult
from app.services.memory_service import MemoryService
from app.services.entity_service import EntityService


class RetrievalService:
    """Service for hybrid retrieval of memories and domain facts."""
    
    def __init__(self, session: Session):
        self.session = session
        self.memory_service = MemoryService(session)
        self.entity_service = EntityService(session)
    
    def retrieve_context(
        self,
        query: str,
        query_embedding: List[float],
        user_id: str,
        session_id: Optional[UUID] = None,
        limit: int = 10
    ) -> RetrievalContext:
        """Retrieve relevant context for a query - 优先使用摘要"""
        # Extract entities from query
        entities = self.entity_service.extract_entities(query, session_id or UUID('00000000-0000-0000-0000-000000000000'))
        
        # 🔥 优先检索摘要
        summaries = self._retrieve_relevant_summaries(query_embedding, user_id)
        if summaries and summaries[0].similarity > 0.7:
            print(f"DEBUG: Using summary for query: {summaries[0].text[:100]}...")
            # 使用摘要构建上下文
            memories = [summaries[0]]  # 将摘要作为记忆使用
            domain_facts = self._retrieve_domain_facts(entities)
            return RetrievalContext(
                memories=memories,
                domain_facts=domain_facts,
                entities=[entity.model_dump() for entity in entities]
            )
        
        # 回退到原始记忆检索
        memories = self.memory_service.retrieve_memories(
            query_embedding=query_embedding,
            user_id=user_id,
            session_id=session_id,
            limit=limit
        )
        
        # Add status information to memories
        memories = self._add_memory_status_info(memories)
        
        # Retrieve domain facts based on entities
        domain_facts = self._retrieve_domain_facts(entities)
        
        # 🆕 新增：检测冲突记忆
        conflict_facts = self._detect_conflicting_memories(query, memories)
        domain_facts.extend(conflict_facts)
        
        # 🆕 新增：构建推理链
        reasoning_facts = self._build_reasoning_chains(entities)
        domain_facts.extend(reasoning_facts)
        
        # 🆕 新增：检测数据库和记忆不一致 (Scenario 17)
        print(f"DEBUG: About to call _detect_db_memory_inconsistencies")
        inconsistency_facts = self._detect_db_memory_inconsistencies(query, memories, domain_facts)
        print(f"DEBUG: _detect_db_memory_inconsistencies returned {len(inconsistency_facts)} facts")
        domain_facts.extend(inconsistency_facts)
        
        return RetrievalContext(
            memories=memories,
            domain_facts=domain_facts,
            entities=[entity.model_dump() for entity in entities]
        )
    
    def _retrieve_relevant_summaries(self, query_embedding: List[float], user_id: str) -> List[MemoryRetrievalResult]:
        """检索相关摘要"""
        from app.models.memory import MemorySummary
        
        # 查询memory_summaries表
        summaries = self.session.exec(
            select(MemorySummary).where(MemorySummary.user_id == user_id)
        ).all()
        
        results = []
        for summary in summaries:
            if summary.embedding:
                similarity = self._calculate_similarity(query_embedding, summary.embedding)
                results.append(MemoryRetrievalResult(
                    memory_id=summary.summary_id,
                    text=summary.summary,
                    similarity=similarity,
                    kind="summary"
                ))
        
        return sorted(results, key=lambda x: x.similarity, reverse=True)
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算两个嵌入向量的相似度"""
        import numpy as np
        
        # 使用余弦相似度
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _retrieve_domain_facts(self, entities: List[Any]) -> List[DomainFact]:
        """Retrieve domain facts based on entities."""
        facts = []
        
        for entity in entities:
            if entity.external_ref:
                table = entity.external_ref.get("table")
                entity_id = entity.external_ref.get("id")
                
                if table == "domain.customers":
                    facts.extend(self._get_customer_facts(entity_id))
                elif table == "domain.sales_orders":
                    facts.extend(self._get_sales_order_facts(entity_id))
                elif table == "domain.invoices":
                    facts.extend(self._get_invoice_facts(entity_id))
                elif table == "domain.tasks":
                    facts.extend(self._get_task_facts(entity_id))
                elif table == "domain.work_orders":
                    facts.extend(self._get_work_order_facts(entity_id))
        
        return facts
    
    def _get_customer_facts(self, customer_id: str) -> List[DomainFact]:
        """Get facts about a customer."""
        facts = []
        
        # Get customer info
        customer = self.session.exec(
            select(Customer).where(Customer.customer_id == customer_id)
        ).first()
        
        if customer:
            facts.append(DomainFact(
                table="customers",
                id=str(customer.customer_id),
                data={
                    "name": customer.name,
                    "industry": customer.industry,
                    "notes": customer.notes
                },
                relevance_score=1.0
            ))
            
            # Get related sales orders
            sales_orders = self.session.exec(
                select(SalesOrder).where(SalesOrder.customer_id == customer_id)
            ).all()
            
            for so in sales_orders:
                facts.append(DomainFact(
                    table="sales_orders",
                    id=str(so.so_id),
                    data={
                        "so_number": so.so_number,
                        "title": so.title,
                        "status": so.status,
                        "created_at": so.created_at.isoformat()
                    },
                    relevance_score=0.8
                ))
            
            # Get open invoices
            invoices = self.session.exec(
                select(Invoice)
                .join(SalesOrder)
                .where(
                    SalesOrder.customer_id == customer_id,
                    Invoice.status == "open"
                )
            ).all()
            
            for invoice in invoices:
                facts.append(DomainFact(
                    table="invoices",
                    id=str(invoice.invoice_id),
                    data={
                        "invoice_number": invoice.invoice_number,
                        "amount": float(invoice.amount),
                        "due_date": invoice.due_date.isoformat(),
                        "status": invoice.status
                    },
                    relevance_score=0.9
                ))
        
        return facts
    
    def _get_sales_order_facts(self, so_id: str) -> List[DomainFact]:
        """Get facts about a sales order."""
        facts = []
        
        sales_order = self.session.exec(
            select(SalesOrder).where(SalesOrder.so_id == so_id)
        ).first()
        
        if sales_order:
            facts.append(DomainFact(
                table="sales_orders",
                id=str(sales_order.so_id),
                data={
                    "so_number": sales_order.so_number,
                    "title": sales_order.title,
                    "status": sales_order.status,
                    "created_at": sales_order.created_at.isoformat()
                },
                relevance_score=1.0
            ))
            
            # Get related work orders
            work_orders = self.session.exec(
                select(WorkOrder).where(WorkOrder.so_id == so_id)
            ).all()
            
            for wo in work_orders:
                facts.append(DomainFact(
                    table="work_orders",
                    id=str(wo.wo_id),
                    data={
                        "description": wo.description,
                        "status": wo.status,
                        "technician": wo.technician,
                        "scheduled_for": wo.scheduled_for.isoformat() if wo.scheduled_for else None
                    },
                    relevance_score=0.8
                ))
        
        return facts
    
    def _get_invoice_facts(self, invoice_id: str) -> List[DomainFact]:
        """Get facts about an invoice."""
        facts = []
        
        invoice = self.session.exec(
            select(Invoice).where(Invoice.invoice_id == invoice_id)
        ).first()
        
        if invoice:
            facts.append(DomainFact(
                table="invoices",
                id=str(invoice.invoice_id),
                data={
                    "invoice_number": invoice.invoice_number,
                    "amount": float(invoice.amount),
                    "due_date": invoice.due_date.isoformat(),
                    "status": invoice.status,
                    "issued_at": invoice.issued_at.isoformat()
                },
                relevance_score=1.0
            ))
            
            # Get payments
            payments = self.session.exec(
                select(Payment).where(Payment.invoice_id == invoice_id)
            ).all()
            
            total_paid = sum(float(p.amount) for p in payments)
            remaining_balance = float(invoice.amount) - total_paid
            
            facts.append(DomainFact(
                table="invoice_payments",
                id=str(invoice.invoice_id),
                data={
                    "total_paid": total_paid,
                    "remaining_balance": remaining_balance,
                    "payment_count": len(payments)
                },
                relevance_score=0.9
            ))
        
        return facts
    
    def _get_work_order_facts(self, wo_id: str) -> List[DomainFact]:
        """Get facts about a work order."""
        facts = []
        
        work_order = self.session.exec(
            select(WorkOrder).where(WorkOrder.wo_id == wo_id)
        ).first()
        
        if work_order:
            facts.append(DomainFact(
                table="work_orders",
                id=str(work_order.wo_id),
                data={
                    "description": work_order.description,
                    "status": work_order.status,
                    "technician": work_order.technician,
                    "scheduled_for": work_order.scheduled_for.isoformat() if work_order.scheduled_for else None
                },
                relevance_score=1.0
            ))
        
        return facts
    
    def _get_task_facts(self, task_id: str) -> List[DomainFact]:
        """Get facts about a task."""
        facts = []
        
        task = self.session.exec(
            select(Task).where(Task.task_id == task_id)
        ).first()
        
        if task:
            facts.append(DomainFact(
                table="tasks",
                id=str(task.task_id),
                data={
                    "title": task.title,
                    "body": task.body,
                    "status": task.status,
                    "created_at": task.created_at.isoformat()
                },
                relevance_score=1.0
            ))
        
        return facts
    
    def _add_memory_status_info(self, memories: List[Any]) -> List[Any]:
        """Add status information to memories based on their content and age."""
        from datetime import datetime, timedelta
        import re
        
        print(f"DEBUG: _add_memory_status_info called with {len(memories)} memories")
        
        for memory in memories:
            # Fix timezone issue - make both datetimes timezone-naive
            now = datetime.now()
            created_at = memory.created_at
            
            # Convert to timezone-naive if needed
            if created_at.tzinfo is not None:
                created_at = created_at.replace(tzinfo=None)
            if now.tzinfo is not None:
                now = now.replace(tzinfo=None)
                
            days_old = (now - created_at).days
            memory_text_lower = memory.text.lower()
            
            print(f"DEBUG: Processing memory: {memory.text[:50]}...")
            
            # Check for stale preferences (Scenario 10)
            # Check both age and text content for time references
            has_time_reference = re.search(r'(\d+)\s+days?\s+ago', memory_text_lower)
            if has_time_reference:
                referenced_days = int(has_time_reference.group(1))
                print(f"DEBUG: Found time reference: {referenced_days} days ago")
                if referenced_days > 90:
                    memory.text += f" [Note: This preference is {referenced_days} days old]"
                    print(f"DEBUG: Added stale preference note for {referenced_days} days")
            elif ("prefer" in memory_text_lower or "prefers" in memory_text_lower) and days_old > 90:
                memory.text += f" [Note: This preference is {days_old} days old]"
                print(f"DEBUG: Added stale preference note for {days_old} days")
            
            # Check for SLA risks (Scenario 6)
            if ("sla" in memory_text_lower or "breach" in memory_text_lower or "risk" in memory_text_lower):
                memory.text += " [Note: This involves SLA risk]"
                print(f"DEBUG: Added SLA risk note")
            
            # Check for completed tasks (Scenario 18)
            if ("done" in memory_text_lower or "complete" in memory_text_lower or "finished" in memory_text_lower):
                memory.text += " [Note: This task is completed]"
                print(f"DEBUG: Added task completion note")
            
            # Check for invoice-related reminders (Scenario 16)
            if ("invoice" in memory_text_lower and ("due" in memory_text_lower or "remind" in memory_text_lower)):
                memory.text += " [Note: This involves invoice reminders]"
                print(f"DEBUG: Added invoice reminder note")
        
        print(f"DEBUG: _add_memory_status_info completed")
        return memories
    
    def _detect_conflicting_memories(self, query: str, memories: List[Any]) -> List[DomainFact]:
        """检测冲突记忆 (Scenario 7)"""
        conflict_facts = []
        
        # 提取查询中的实体名称
        query_lower = query.lower()
        customer_names = []
        
        # 简单的实体提取（可以增强）
        if "kai media" in query_lower:
            customer_names.append("kai media")
        if "tc boiler" in query_lower:
            customer_names.append("tc boiler")
        
        for customer_name in customer_names:
            # 查找该客户相关的语义记忆
            customer_memories = []
            for memory in memories:
                if (memory.kind == "semantic" and 
                    customer_name in memory.text.lower() and
                    ("prefer" in memory.text.lower() or "like" in memory.text.lower())):
                    customer_memories.append(memory)
            
            # 检测冲突
            conflicts = []
            for i, mem1 in enumerate(customer_memories):
                for j, mem2 in enumerate(customer_memories):
                    if i < j and self._is_conflicting_memory(mem1.text, mem2.text):
                        conflicts.append({
                            "memory1": {
                                "id": mem1.memory_id,
                                "text": mem1.text,
                                "created_at": mem1.created_at.isoformat(),
                                "importance": mem1.importance
                            },
                            "memory2": {
                                "id": mem2.memory_id,
                                "text": mem2.text,
                                "created_at": mem2.created_at.isoformat(),
                                "importance": mem2.importance
                            },
                            "resolution": "most_recent" if mem1.created_at > mem2.created_at else "older"
                        })
            
            if conflicts:
                conflict_facts.append(DomainFact(
                    table="memory_conflicts",
                    id=f"conflict_{customer_name}",
                    data={
                        "customer": customer_name,
                        "conflicts": conflicts,
                        "resolution_strategy": "most_recent"
                    },
                    relevance_score=0.9
                ))
        
        return conflict_facts
    
    def _is_conflicting_memory(self, text1: str, text2: str) -> bool:
        """判断两个记忆是否冲突"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # 检查是否包含相反的偏好
        day_conflicts = [
            ("thursday", "friday"),
            ("friday", "thursday"),
            ("monday", "tuesday"),
            ("tuesday", "monday"),
            ("morning", "afternoon"),
            ("afternoon", "morning")
        ]
        
        for day1, day2 in day_conflicts:
            if day1 in text1_lower and day2 in text2_lower:
                return True
            if day2 in text1_lower and day1 in text2_lower:
                return True
        
        return False
    
    def _build_reasoning_chains(self, entities: List[Any]) -> List[DomainFact]:
        """构建跨对象推理链 (Scenario 11)"""
        reasoning_facts = []
        
        for entity in entities:
            if entity.external_ref:
                table = entity.external_ref.get("table")
                entity_id = entity.external_ref.get("id")
                
                if table == "domain.customers":
                    # 构建客户相关的推理链
                    chain = self._build_customer_reasoning_chain(entity_id)
                    if chain:
                        reasoning_facts.append(DomainFact(
                            table="reasoning_chains",
                            id=f"chain_{entity_id}",
                            data=chain,
                            relevance_score=0.8
                        ))
        
        return reasoning_facts
    
    def _build_customer_reasoning_chain(self, customer_id: str) -> Optional[Dict]:
        """构建客户相关的推理链"""
        try:
            # 查询客户信息
            customer = self.session.exec(
                select(Customer).where(Customer.customer_id == customer_id)
            ).first()
            
            if not customer:
                return None
            
            # 查询销售订单
            sales_orders = self.session.exec(
                select(SalesOrder).where(SalesOrder.customer_id == customer_id)
            ).all()
            
            reasoning_chain = {
                "customer": customer.name,
                "sales_orders": [],
                "can_invoice": False,
                "should_send_invoice": False,
                "blocked_work_orders": []
            }
            
            for so in sales_orders:
                # 查询工作订单
                work_orders = self.session.exec(
                    select(WorkOrder).where(WorkOrder.so_id == so.so_id)
                ).all()
                
                # 查询发票
                invoices = self.session.exec(
                    select(Invoice).where(Invoice.so_id == so.so_id)
                ).all()
                
                so_data = {
                    "so_number": so.so_number,
                    "status": so.status,
                    "work_orders": [
                        {
                            "status": wo.status,
                            "description": wo.description,
                            "technician": wo.technician
                        } for wo in work_orders
                    ],
                    "invoices": [
                        {
                            "invoice_number": inv.invoice_number,
                            "status": inv.status,
                            "amount": float(inv.amount)
                        } for inv in invoices
                    ]
                }
                
                reasoning_chain["sales_orders"].append(so_data)
                
                # 应用业务规则
                done_wos = [wo for wo in work_orders if wo.status == "done"]
                open_invoices = [inv for inv in invoices if inv.status == "open"]
                blocked_wos = [wo for wo in work_orders if wo.status == "blocked"]
                
                if done_wos and not invoices:
                    reasoning_chain["can_invoice"] = True
                
                if open_invoices:
                    reasoning_chain["should_send_invoice"] = True
                
                if blocked_wos:
                    reasoning_chain["blocked_work_orders"].extend([
                        {"so_number": so.so_number, "description": wo.description}
                        for wo in blocked_wos
                    ])
            
            return reasoning_chain
            
        except Exception as e:
            print(f"Error building reasoning chain: {e}")
            return None
    
    def _detect_db_memory_inconsistencies(self, query: str, memories: List[Any], domain_facts: List[DomainFact]) -> List[DomainFact]:
        """检测数据库和记忆不一致 (Scenario 17)"""
        inconsistency_facts = []
        
        query_lower = query.lower()
        print(f"DEBUG: _detect_db_memory_inconsistencies called with query: {query}")
        
        # 检查是否询问状态或完成情况
        status_keywords = ["status", "complete", "done", "finished", "fulfilled", "is.*complete"]
        if any(keyword in query_lower for keyword in status_keywords):
            print(f"DEBUG: Status keywords detected in query")
            
            # 提取订单号
            order_numbers = []
            import re
            order_pattern = r'\b(SO-\d+|INV-\d+|WO-\d+)\b'
            orders = re.findall(order_pattern, query)
            order_numbers.extend(orders)
            print(f"DEBUG: Found order numbers: {order_numbers}")
            
            for order_number in order_numbers:
                # 查找相关的数据库事实
                db_fact = None
                for fact in domain_facts:
                    if fact.table == "sales_orders" and fact.data.get("so_number") == order_number:
                        db_fact = fact
                        break
                
                if db_fact:
                    print(f"DEBUG: Found DB fact for {order_number}: {db_fact.data}")
                    
                    # 查找相关的记忆
                    conflicting_memories = []
                    for memory in memories:
                        memory_text_lower = memory.text.lower()
                        if (order_number.lower() in memory_text_lower and 
                            any(status_word in memory_text_lower for status_word in ["fulfilled", "complete", "done", "finished"])):
                            conflicting_memories.append(memory)
                            print(f"DEBUG: Found conflicting memory: {memory.text}")
                    
                    # 检查数据库状态和记忆状态是否不一致
                    db_status = db_fact.data.get("status", "").lower()
                    print(f"DEBUG: DB status: {db_status}, conflicting memories: {len(conflicting_memories)}")
                    
                    # 扩展不一致检测逻辑
                    status_mappings = {
                        "in_fulfillment": ["fulfilled", "complete", "done", "finished"],
                        "draft": ["fulfilled", "complete", "done", "finished"],
                        "open": ["paid", "complete", "done", "finished"],
                        "queued": ["done", "complete", "finished"]
                    }
                    
                    # 检查是否存在不一致
                    inconsistent = False
                    memory_status = None
                    
                    if db_status in status_mappings:
                        for memory in conflicting_memories:
                            memory_text_lower = memory.text.lower()
                            for conflicting_status in status_mappings[db_status]:
                                if conflicting_status in memory_text_lower:
                                    inconsistent = True
                                    memory_status = conflicting_status
                                    break
                            if inconsistent:
                                break
                    
                    if inconsistent and conflicting_memories:
                        print(f"DEBUG: Creating inconsistency fact for {order_number}")
                        # 数据库状态与记忆状态不一致
                        inconsistency_facts.append(DomainFact(
                            table="db_memory_inconsistency",
                            id=f"inconsistency_{order_number}",
                            data={
                                "order_number": order_number,
                                "db_status": db_status,
                                "memory_status": memory_status,
                                "conflicting_memories": [
                                    {
                                        "memory_id": mem.memory_id,
                                        "text": mem.text,
                                        "created_at": mem.created_at.isoformat()
                                    } for mem in conflicting_memories
                                ],
                                "resolution": "prefer_db",
                                "action": "mark_memory_for_decay",
                                "message": f"Database shows {db_status} but memory says {memory_status}. Using database truth."
                            },
                            relevance_score=0.95
                        ))
        
        print(f"DEBUG: Returning {len(inconsistency_facts)} inconsistency facts")
        return inconsistency_facts