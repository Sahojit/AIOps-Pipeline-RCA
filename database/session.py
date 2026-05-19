"""
database/session.py
===================
SQLAlchemy engine and session factory.

Design decisions:
- Engine is created once at module import (singleton pattern)
- Sessions are created per-request via get_db() generator
- Connection pooling is configured via settings (pool_size, max_overflow)
- The generator pattern ensures sessions are ALWAYS closed, even on exceptions

WHY a session generator?
- FastAPI's Depends(get_db) injects a session per request
- The generator yields the session, then closes it in the finally block
- This prevents connection leaks — the #1 cause of database outages in
  production Python applications
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Engine — one per process, connection-pooled
# ---------------------------------------------------------------------------
engine = create_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    echo=settings.db_echo,  # Set True in .env to log all SQL queries
    pool_pre_ping=True,      # Detect stale connections before using them
)

# ---------------------------------------------------------------------------
# Session factory — creates new sessions bound to the engine
# ---------------------------------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Declarative base — all table models inherit from this
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# ---------------------------------------------------------------------------
# Session dependency — used by FastAPI's Depends()
# ---------------------------------------------------------------------------
def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session, ensuring it's closed after use.

    Usage in FastAPI:
        @app.post("/endpoint")
        def handler(db: Session = Depends(get_db)):
            db.query(...)

    The session is committed by the caller. If an exception occurs,
    the session is rolled back and closed.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_tables() -> None:
    """
    Create all tables defined in the ORM models.

    Call this at application startup or via a migration script.
    In production, use Alembic migrations instead of create_all().
    """
    logger.info("Creating database tables")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
