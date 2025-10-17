"""Create memory schema and tables

Revision ID: 002_create_memory_schema
Revises: 001_create_domain_schema
Create Date: 2024-01-15 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002_create_memory_schema'
down_revision = '001_create_domain_schema'
branch_labels = None
depends_on = None


def upgrade():
    # Create app schema
    op.execute('CREATE SCHEMA IF NOT EXISTS app')
    
    # Create chat_events table
    op.create_table(
        'chat_events',
        sa.Column('event_id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.text('now()')),
        sa.CheckConstraint("role IN ('user','assistant','system')", name='check_chat_event_role'),
        schema='app'
    )
    
    # Create entities table
    op.create_table(
        'entities',
        sa.Column('entity_id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('type', sa.Text(), nullable=False),
        sa.Column('source', sa.Text(), nullable=False),
        sa.Column('external_ref', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.text('now()')),
        sa.CheckConstraint("source IN ('message','db')", name='check_entity_source'),
        schema='app'
    )
    
    # Create memories table
    op.create_table(
        'memories',
        sa.Column('memory_id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),  # Will be converted to vector
        sa.Column('importance', sa.Float(), nullable=False, default=0.5),
        sa.Column('ttl_days', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.text('now()')),
        sa.CheckConstraint("kind IN ('episodic','semantic','profile','commitment','todo')", name='check_memory_kind'),
        sa.CheckConstraint("importance >= 0 AND importance <= 1", name='check_memory_importance'),
        schema='app'
    )
    
    # Create memory_summaries table
    op.create_table(
        'memory_summaries',
        sa.Column('summary_id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.Column('session_window', sa.Integer(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),  # Will be converted to vector
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.text('now()')),
        schema='app'
    )
    
    # Convert array columns to vector type
    op.execute('ALTER TABLE app.memories ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector')
    op.execute('ALTER TABLE app.memory_summaries ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector')
    
    # Create vector indexes
    op.execute('CREATE INDEX idx_memories_embedding ON app.memories USING ivfflat (embedding vector_cosine_ops)')
    op.execute('CREATE INDEX idx_memory_summaries_embedding ON app.memory_summaries USING ivfflat (embedding vector_cosine_ops)')
    
    # Create regular indexes for performance
    op.create_index('idx_chat_events_session_id', 'chat_events', ['session_id'], schema='app')
    op.create_index('idx_entities_session_id', 'entities', ['session_id'], schema='app')
    op.create_index('idx_entities_name', 'entities', ['name'], schema='app')
    op.create_index('idx_memories_session_id', 'memories', ['session_id'], schema='app')
    op.create_index('idx_memories_kind', 'memories', ['kind'], schema='app')
    op.create_index('idx_memory_summaries_user_id', 'memory_summaries', ['user_id'], schema='app')


def downgrade():
    # Drop indexes
    op.drop_index('idx_memory_summaries_user_id', 'memory_summaries', schema='app')
    op.drop_index('idx_memories_kind', 'memories', schema='app')
    op.drop_index('idx_memories_session_id', 'memories', schema='app')
    op.drop_index('idx_entities_name', 'entities', schema='app')
    op.drop_index('idx_entities_session_id', 'entities', schema='app')
    op.drop_index('idx_chat_events_session_id', 'chat_events', schema='app')
    
    # Drop vector indexes
    op.execute('DROP INDEX IF EXISTS idx_memory_summaries_embedding')
    op.execute('DROP INDEX IF EXISTS idx_memories_embedding')
    
    # Drop tables
    op.drop_table('memory_summaries', schema='app')
    op.drop_table('memories', schema='app')
    op.drop_table('entities', schema='app')
    op.drop_table('chat_events', schema='app')
    
    # Drop app schema
    op.execute('DROP SCHEMA IF EXISTS app CASCADE')
