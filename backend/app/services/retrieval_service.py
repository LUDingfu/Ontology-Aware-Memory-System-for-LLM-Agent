"""Retrieval service for hybrid search."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlmodel import Session, select, text

from app.models.domain import Customer, SalesOrder, Invoice, Task, Payment, WorkOrder
from app.models.chat import DomainFact, RetrievalContext
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
        """Retrieve relevant context for a query."""
        # Extract entities from query
        entities = self.entity_service.extract_entities(query, session_id or UUID('00000000-0000-0000-0000-000000000000'))
        
        # Retrieve memories
        memories = self.memory_service.retrieve_memories(
            query_embedding=query_embedding,
            user_id=user_id,
            session_id=session_id,
            limit=limit
        )
        
        # Retrieve domain facts based on entities
        domain_facts = self._retrieve_domain_facts(entities)
        
        return RetrievalContext(
            memories=memories,
            domain_facts=domain_facts,
            entities=[entity.model_dump() for entity in entities]
        )
    
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
