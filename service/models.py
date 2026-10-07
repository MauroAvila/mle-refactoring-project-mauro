"""SQLAlchemy ORM Models for the King County Housing Web Service.

This module defines the database schema using SQLAlchemy 2.0 typed mappings.
Each class represents a physical database table and declares its columns,
data types, and constraints.
"""

from sqlalchemy import Float, Integer
from sqlalchemy.orm import Mapped, mapped_column

from service.database import Base


class Property(Base):
    """SQLAlchemy ORM model representing the physical 'properties' table."""

    __tablename__ = "properties"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    bedrooms: Mapped[int] = mapped_column(Integer, nullable=False)
    bathrooms: Mapped[float] = mapped_column(Float, nullable=False)
    sqft_living: Mapped[int] = mapped_column(Integer, nullable=False)
    sqft_lot: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
