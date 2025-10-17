"""Create domain schema and tables

Revision ID: 001_create_domain_schema
Revises: 
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_create_domain_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create domain schema
    op.execute('CREATE SCHEMA IF NOT EXISTS domain')
    
    # Create customers table
    op.create_table(
        'customers',
        sa.Column('customer_id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('industry', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        schema='domain'
    )
    
    # Create sales_orders table
    op.create_table(
        'sales_orders',
        sa.Column('so_id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('customer_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('so_number', sa.Text(), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.text('now()')),
        sa.ForeignKeyConstraint(['customer_id'], ['domain.customers.customer_id']),
        sa.CheckConstraint("status IN ('draft','approved','in_fulfillment','fulfilled','cancelled')", name='check_sales_order_status'),
        schema='domain'
    )
    
    # Create unique index on so_number
    op.create_index('idx_sales_orders_so_number', 'sales_orders', ['so_number'], unique=True, schema='domain')
    
    # Create work_orders table
    op.create_table(
        'work_orders',
        sa.Column('wo_id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('so_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('technician', sa.Text(), nullable=True),
        sa.Column('scheduled_for', sa.Date(), nullable=True),
        sa.ForeignKeyConstraint(['so_id'], ['domain.sales_orders.so_id']),
        sa.CheckConstraint("status IN ('queued','in_progress','blocked','done')", name='check_work_order_status'),
        schema='domain'
    )
    
    # Create invoices table
    op.create_table(
        'invoices',
        sa.Column('invoice_id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('so_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('invoice_number', sa.Text(), nullable=False),
        sa.Column('amount', sa.Numeric(12, 2), nullable=False),
        sa.Column('due_date', sa.Date(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('issued_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.text('now()')),
        sa.ForeignKeyConstraint(['so_id'], ['domain.sales_orders.so_id']),
        sa.CheckConstraint("status IN ('open','paid','void')", name='check_invoice_status'),
        schema='domain'
    )
    
    # Create unique index on invoice_number
    op.create_index('idx_invoices_invoice_number', 'invoices', ['invoice_number'], unique=True, schema='domain')
    
    # Create payments table
    op.create_table(
        'payments',
        sa.Column('payment_id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('invoice_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('amount', sa.Numeric(12, 2), nullable=False),
        sa.Column('method', sa.Text(), nullable=True),
        sa.Column('paid_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.text('now()')),
        sa.ForeignKeyConstraint(['invoice_id'], ['domain.invoices.invoice_id']),
        schema='domain'
    )
    
    # Create tasks table
    op.create_table(
        'tasks',
        sa.Column('task_id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('customer_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('body', sa.Text(), nullable=True),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.text('now()')),
        sa.ForeignKeyConstraint(['customer_id'], ['domain.customers.customer_id']),
        sa.CheckConstraint("status IN ('todo','doing','done')", name='check_task_status'),
        schema='domain'
    )


def downgrade():
    # Drop tables in reverse order
    op.drop_table('tasks', schema='domain')
    op.drop_table('payments', schema='domain')
    op.drop_table('invoices', schema='domain')
    op.drop_table('work_orders', schema='domain')
    op.drop_table('sales_orders', schema='domain')
    op.drop_table('customers', schema='domain')
    
    # Drop domain schema
    op.execute('DROP SCHEMA IF EXISTS domain CASCADE')
