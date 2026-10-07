"""Pydantic Schemas for the King County Housing Web Service.

This module defines the data validation contracts for incoming request payloads
and outgoing HTTP response models using Pydantic v2.
"""

from pydantic import BaseModel, ConfigDict, Field


class PropertyBase(BaseModel):
    """Base schema declaring shared fields and validation rules for properties."""

    bedrooms: int = Field(
        ..., ge=1, le=50, description="Number of bedrooms in the property"
    )
    bathrooms: float = Field(
        ..., ge=0.5, le=20.0, description="Number of bathrooms in the property"
    )
    sqft_living: int = Field(
        ..., gt=0, description="Interior living space in square feet"
    )
    sqft_lot: int = Field(
        ..., gt=0, description="Total lot size in square feet"
    )
    price: float = Field(
        ..., gt=0.0, description="Sale price of the property in USD"
    )


class PropertyCreate(PropertyBase):
    """Schema for validating creation payloads (POST requests)."""

    pass


class PropertyUpdate(PropertyBase):
    """Schema for validating update payloads (PUT requests)."""

    pass


class PropertyOut(PropertyBase):
    """Schema for serializing outgoing property response objects (JSON output)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
