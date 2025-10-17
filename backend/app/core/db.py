from sqlmodel import Session, create_engine

from app.core.config import settings

engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))


def init_db(session: Session) -> None:
    """Initialize database with required schemas and extensions."""
    # Tables should be created with Alembic migrations
    # This function can be used for any additional initialization if needed
    pass
