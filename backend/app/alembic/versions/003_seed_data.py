"""Seed initial data

Revision ID: 003_seed_data
Revises: 002_create_memory_schema
Create Date: 2024-01-15 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_seed_data'
down_revision = '002_create_memory_schema'
branch_labels = None
depends_on = None


def upgrade():
    # Insert sample customers
    op.execute("""
        INSERT INTO domain.customers (customer_id, name, industry, notes) VALUES
        ('550e8400-e29b-41d4-a716-446655440001', 'Gai Media', 'Entertainment', 'Music production company'),
        ('550e8400-e29b-41d4-a716-446655440002', 'PC Boiler', 'Industrial', 'Industrial boiler manufacturer'),
        ('550e8400-e29b-41d4-a716-446655440003', 'Kai Media', 'Entertainment', 'Digital media company'),
        ('550e8400-e29b-41d4-a716-446655440004', 'Kai Media Europe', 'Entertainment', 'European division'),
        ('550e8400-e29b-41d4-a716-446655440005', 'TC Boiler', 'Industrial', 'Thermal control systems')
    """)
    
    # Insert sample sales orders
    op.execute("""
        INSERT INTO domain.sales_orders (so_id, customer_id, so_number, title, status, created_at) VALUES
        ('660e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', 'SO-1001', 'Album Fulfillment', 'in_fulfillment', '2024-01-10 09:00:00+00'),
        ('660e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440002', 'SO-2002', 'On-site repair', 'approved', '2024-01-12 14:30:00+00'),
        ('660e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440003', 'SO-3003', 'Digital Content Package', 'fulfilled', '2024-01-08 11:15:00+00'),
        ('660e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440005', 'SO-4004', 'Boiler Maintenance', 'draft', '2024-01-15 08:45:00+00')
    """)
    
    # Insert sample work orders
    op.execute("""
        INSERT INTO domain.work_orders (wo_id, so_id, description, status, technician, scheduled_for) VALUES
        ('770e8400-e29b-41d4-a716-446655440001', '660e8400-e29b-41d4-a716-446655440001', 'Pick-pack albums', 'queued', 'Alex', '2024-01-22'),
        ('770e8400-e29b-41d4-a716-446655440002', '660e8400-e29b-41d4-a716-446655440002', 'Replace valve', 'in_progress', 'Bob', '2024-01-20'),
        ('770e8400-e29b-41d4-a716-446655440003', '660e8400-e29b-41d4-a716-446655440003', 'Digital packaging', 'done', 'Carol', '2024-01-18'),
        ('770e8400-e29b-41d4-a716-446655440004', '660e8400-e29b-41d4-a716-446655440004', 'Boiler inspection', 'queued', 'Dave', '2024-01-25')
    """)
    
    # Insert sample invoices
    op.execute("""
        INSERT INTO domain.invoices (invoice_id, so_id, invoice_number, amount, due_date, status, issued_at) VALUES
        ('880e8400-e29b-41d4-a716-446655440001', '660e8400-e29b-41d4-a716-446655440001', 'INV-1009', 1200.00, '2024-09-30', 'open', '2024-01-10 10:00:00+00'),
        ('880e8400-e29b-41d4-a716-446655440002', '660e8400-e29b-41d4-a716-446655440002', 'INV-2010', 850.00, '2024-02-15', 'open', '2024-01-12 15:00:00+00'),
        ('880e8400-e29b-41d4-a716-446655440003', '660e8400-e29b-41d4-a716-446655440003', 'INV-3011', 2100.00, '2024-02-08', 'paid', '2024-01-08 12:00:00+00'),
        ('880e8400-e29b-41d4-a716-446655440004', '660e8400-e29b-41d4-a716-446655440004', 'INV-4012', 1500.00, '2024-02-20', 'open', '2024-01-15 09:00:00+00')
    """)
    
    # Insert sample payments
    op.execute("""
        INSERT INTO domain.payments (payment_id, invoice_id, amount, method, paid_at) VALUES
        ('990e8400-e29b-41d4-a716-446655440001', '880e8400-e29b-41d4-a716-446655440003', 2100.00, 'ACH', '2024-01-15 14:30:00+00'),
        ('990e8400-e29b-41d4-a716-446655440002', '880e8400-e29b-41d4-a716-446655440001', 600.00, 'Credit Card', '2024-01-20 10:15:00+00')
    """)
    
    # Insert sample tasks
    op.execute("""
        INSERT INTO domain.tasks (task_id, customer_id, title, body, status, created_at) VALUES
        ('aa0e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', 'Investigate shipping SLA for Gai Media', 'Check delivery timeframes and customer preferences', 'todo', '2024-01-05 09:00:00+00'),
        ('aa0e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440002', 'Schedule maintenance visit', 'Coordinate with customer for boiler maintenance', 'doing', '2024-01-12 11:30:00+00'),
        ('aa0e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440003', 'Follow up on payment', 'Contact customer about overdue invoice', 'todo', '2024-01-14 16:45:00+00'),
        ('aa0e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440005', 'Prepare quote for new system', 'Create proposal for thermal control upgrade', 'todo', '2024-01-15 08:00:00+00')
    """)


def downgrade():
    # Delete all seeded data
    op.execute('DELETE FROM domain.tasks')
    op.execute('DELETE FROM domain.payments')
    op.execute('DELETE FROM domain.invoices')
    op.execute('DELETE FROM domain.work_orders')
    op.execute('DELETE FROM domain.sales_orders')
    op.execute('DELETE FROM domain.customers')
