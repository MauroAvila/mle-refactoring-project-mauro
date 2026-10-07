import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Retrieve the database connection string from system environment variables
DATABASE_URL = os.getenv("DB_CONN")

if DATABASE_URL is None:
    raise ValueError("DB_CONN environment variable is required.")

# Initialize the SQLAlchemy database engine
engine = create_engine(DATABASE_URL)

# Create a session factory bound to the database engine
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Base class shared by all SQLAlchemy ORM models in the service."""

    pass


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for a single HTTP request and close it afterwards."""
    with SessionLocal() as session:
        yield session
