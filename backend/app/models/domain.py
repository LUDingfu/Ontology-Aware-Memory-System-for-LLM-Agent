"""Domain models for business entities."""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

from sqlmodel import Field, Relationship, SQLModel


class Customer(SQLModel, table=True):
    """Customer model representing business customers."""
    
    __tablename__ = "customers"
    __table_args__ = {"schema": "domain"}
    
    customer_id: UUID = Field(default=None, primary_key=True)
    name: str
    industry: Optional[str] = None
    notes: Optional[str] = None
    
    # Relationships
    sales_orders: list["SalesOrder"] = Relationship(back_populates="customer")
    tasks: list["Task"] = Relationship(back_populates="customer")


class SalesOrder(SQLModel, table=True):
    """Sales order model."""
    
    __tablename__ = "sales_orders"
    __table_args__ = {"schema": "domain"}
    
    so_id: UUID = Field(default=None, primary_key=True)
    customer_id: UUID = Field(foreign_key="domain.customers.customer_id")
    so_number: str = Field(unique=True)
    title: str
    status: str = Field(regex="^(draft|approved|in_fulfillment|fulfilled|cancelled)$")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    customer: Optional[Customer] = Relationship(back_populates="sales_orders")
    work_orders: list["WorkOrder"] = Relationship(back_populates="sales_order")
    invoices: list["Invoice"] = Relationship(back_populates="sales_order")


class WorkOrder(SQLModel, table=True):
    """Work order model."""
    
    __tablename__ = "work_orders"
    __table_args__ = {"schema": "domain"}
    
    wo_id: UUID = Field(default=None, primary_key=True)
    so_id: UUID = Field(foreign_key="domain.sales_orders.so_id")
    description: Optional[str] = None
    status: str = Field(regex="^(queued|in_progress|blocked|done)$")
    technician: Optional[str] = None
    scheduled_for: Optional[date] = None
    
    # Relationships
    sales_order: Optional[SalesOrder] = Relationship(back_populates="work_orders")


class Invoice(SQLModel, table=True):
    """Invoice model."""
    
    __tablename__ = "invoices"
    __table_args__ = {"schema": "domain"}
    
    invoice_id: UUID = Field(default=None, primary_key=True)
    so_id: UUID = Field(foreign_key="domain.sales_orders.so_id")
    invoice_number: str = Field(unique=True)
    amount: Decimal = Field(max_digits=12, decimal_places=2)
    due_date: date
    status: str = Field(regex="^(open|paid|void)$")
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    sales_order: Optional[SalesOrder] = Relationship(back_populates="invoices")
    payments: list["Payment"] = Relationship(back_populates="invoice")


class Payment(SQLModel, table=True):
    """Payment model."""
    
    __tablename__ = "payments"
    __table_args__ = {"schema": "domain"}
    
    payment_id: UUID = Field(default=None, primary_key=True)
    invoice_id: UUID = Field(foreign_key="domain.invoices.invoice_id")
    amount: Decimal = Field(max_digits=12, decimal_places=2)
    method: Optional[str] = None
    paid_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    invoice: Optional[Invoice] = Relationship(back_populates="payments")


class Task(SQLModel, table=True):
    """Task model for support and operational tasks."""
    
    __tablename__ = "tasks"
    __table_args__ = {"schema": "domain"}
    
    task_id: UUID = Field(default=None, primary_key=True)
    customer_id: Optional[UUID] = Field(foreign_key="domain.customers.customer_id")
    title: str
    body: Optional[str] = None
    status: str = Field(regex="^(todo|doing|done)$")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    customer: Optional[Customer] = Relationship(back_populates="tasks")
