"""
Database connection utilities for Archon server services.
"""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# Get database URL from environment or use default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./archon.db"  # Default to SQLite for development
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """Get database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Generator[Session, None, None]:
    """Alias for get_db for backward compatibility."""
    return get_db()