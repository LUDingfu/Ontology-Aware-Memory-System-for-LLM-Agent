"""Entity recognition and linking service."""

import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlmodel import Session, select

from app.models.domain import Customer, SalesOrder, Invoice, Task
from app.models.memory import Entity


class EntityService:
    """Service for entity recognition and linking."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def extract_entities(
        self,
        text: str,
        session_id: UUID
    ) -> List[Entity]:
        """Extract entities from text using NER and rule-based matching."""
        entities = []
        
        # Extract customer names
        customer_entities = self._extract_customer_entities(text, session_id)
        entities.extend(customer_entities)
        
        # Extract order numbers
        order_entities = self._extract_order_entities(text, session_id)
        entities.extend(order_entities)
        
        # Extract invoice numbers
        invoice_entities = self._extract_invoice_entities(text, session_id)
        entities.extend(invoice_entities)
        
        # Extract task references
        task_entities = self._extract_task_entities(text, session_id)
        entities.extend(task_entities)
        
        return entities
    
    def link_entities_to_domain(self, entities: List[Entity]) -> List[Entity]:
        """Link extracted entities to domain database records."""
        linked_entities = []
        
        for entity in entities:
            if entity.type == "customer":
                linked_entity = self._link_customer_entity(entity)
            elif entity.type == "order":
                linked_entity = self._link_order_entity(entity)
            elif entity.type == "invoice":
                linked_entity = self._link_invoice_entity(entity)
            elif entity.type == "task":
                linked_entity = self._link_task_entity(entity)
            else:
                linked_entity = entity
            
            linked_entities.append(linked_entity)
        
        return linked_entities
    
    def _extract_customer_entities(self, text: str, session_id: UUID) -> List[Entity]:
        """Extract customer names from text."""
        entities = []
        
        # Get all customers from database
        customers = self.session.exec(select(Customer)).all()
        
        for customer in customers:
            # Exact match
            if customer.name.lower() in text.lower():
                entity = Entity(
                    session_id=session_id,
                    name=customer.name,
                    type="customer",
                    source="db",
                    external_ref={
                        "table": "domain.customers",
                        "id": str(customer.customer_id)
                    }
                )
                entities.append(entity)
            
            # Fuzzy match for partial names
            elif self._fuzzy_match(customer.name, text):
                entity = Entity(
                    session_id=session_id,
                    name=customer.name,
                    type="customer",
                    source="db",
                    external_ref={
                        "table": "domain.customers",
                        "id": str(customer.customer_id),
                        "confidence": "fuzzy"
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _extract_order_entities(self, text: str, session_id: UUID) -> List[Entity]:
        """Extract sales order numbers from text."""
        entities = []
        
        # Pattern for order numbers (SO-XXXX)
        order_pattern = r'SO-\d{4}'
        matches = re.findall(order_pattern, text, re.IGNORECASE)
        
        for match in matches:
            # Check if order exists in database
            order = self.session.exec(
                select(SalesOrder).where(SalesOrder.so_number == match.upper())
            ).first()
            
            if order:
                entity = Entity(
                    session_id=session_id,
                    name=match,
                    type="order",
                    source="db",
                    external_ref={
                        "table": "domain.sales_orders",
                        "id": str(order.so_id)
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _extract_invoice_entities(self, text: str, session_id: UUID) -> List[Entity]:
        """Extract invoice numbers from text."""
        entities = []
        
        # Pattern for invoice numbers (INV-XXXX)
        invoice_pattern = r'INV-\d{4}'
        matches = re.findall(invoice_pattern, text, re.IGNORECASE)
        
        for match in matches:
            # Check if invoice exists in database
            invoice = self.session.exec(
                select(Invoice).where(Invoice.invoice_number == match.upper())
            ).first()
            
            if invoice:
                entity = Entity(
                    session_id=session_id,
                    name=match,
                    type="invoice",
                    source="db",
                    external_ref={
                        "table": "domain.invoices",
                        "id": str(invoice.invoice_id)
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _extract_task_entities(self, text: str, session_id: UUID) -> List[Entity]:
        """Extract task references from text."""
        entities = []
        
        # Look for task-related keywords
        task_keywords = ['task', 'todo', 'issue', 'problem', 'support']
        
        for keyword in task_keywords:
            if keyword.lower() in text.lower():
                # Find related tasks in database
                tasks = self.session.exec(select(Task)).all()
                
                for task in tasks:
                    if keyword.lower() in task.title.lower() or keyword.lower() in (task.body or "").lower():
                        entity = Entity(
                            session_id=session_id,
                            name=task.title,
                            type="task",
                            source="db",
                            external_ref={
                                "table": "domain.tasks",
                                "id": str(task.task_id)
                            }
                        )
                        entities.append(entity)
        
        return entities
    
    def _link_customer_entity(self, entity: Entity) -> Entity:
        """Link customer entity to domain record."""
        customer = self.session.exec(
            select(Customer).where(Customer.name == entity.name)
        ).first()
        
        if customer:
            entity.external_ref = {
                "table": "domain.customers",
                "id": str(customer.customer_id)
            }
            entity.source = "db"
        
        return entity
    
    def _link_order_entity(self, entity: Entity) -> Entity:
        """Link order entity to domain record."""
        order = self.session.exec(
            select(SalesOrder).where(SalesOrder.so_number == entity.name)
        ).first()
        
        if order:
            entity.external_ref = {
                "table": "domain.sales_orders",
                "id": str(order.so_id)
            }
            entity.source = "db"
        
        return entity
    
    def _link_invoice_entity(self, entity: Entity) -> Entity:
        """Link invoice entity to domain record."""
        invoice = self.session.exec(
            select(Invoice).where(Invoice.invoice_number == entity.name)
        ).first()
        
        if invoice:
            entity.external_ref = {
                "table": "domain.invoices",
                "id": str(invoice.invoice_id)
            }
            entity.source = "db"
        
        return entity
    
    def _link_task_entity(self, entity: Entity) -> Entity:
        """Link task entity to domain record."""
        task = self.session.exec(
            select(Task).where(Task.title == entity.name)
        ).first()
        
        if task:
            entity.external_ref = {
                "table": "domain.tasks",
                "id": str(task.task_id)
            }
            entity.source = "db"
        
        return entity
    
    def _fuzzy_match(self, name: str, text: str, threshold: float = 0.8) -> bool:
        """Check if name fuzzy matches text."""
        name_words = set(name.lower().split())
        text_words = set(text.lower().split())
        
        if not name_words:
            return False
        
        intersection = name_words.intersection(text_words)
        similarity = len(intersection) / len(name_words)
        
        return similarity >= threshold
    
    def get_entities_for_session(self, session_id: UUID) -> List[Entity]:
        """Get all entities for a session."""
        entities = self.session.exec(
            select(Entity).where(Entity.session_id == session_id)
        ).all()
        
        return entities
