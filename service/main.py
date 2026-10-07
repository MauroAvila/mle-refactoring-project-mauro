"""Main FastAPI Application for the King County Housing Web Service.

This module configures the FastAPI instance, initializes database tables,
and implements RESTful HTTP routes for complete CRUD operations on housing properties.
"""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from service import models, schemas
from service.database import Base, engine, get_db

# Create physical database tables if they do not exist
Base.metadata.create_all(bind=engine)

# Initialize FastAPI application instance
app = FastAPI(
    title="King County Housing REST API",
    description="CRUD API for managing and querying housing property records.",
    version="1.0.0",
)

# Type alias for request-scoped database session dependency
DatabaseSession = Annotated[Session, Depends(get_db)]


@app.get("/", status_code=status.HTTP_200_OK)
def root():
    """Health check endpoint confirming API service status."""
    return {"status": "ok", "message": "King County Housing Service is active."}


@app.post(
    "/properties",
    response_model=schemas.PropertyOut,
    status_code=status.HTTP_201_CREATED,
)
def create_property(
    request: schemas.PropertyCreate, db: DatabaseSession
) -> models.Property:
    """Create a new property record in the database."""
    new_property = models.Property(
        bedrooms=request.bedrooms,
        bathrooms=request.bathrooms,
        sqft_living=request.sqft_living,
        sqft_lot=request.sqft_lot,
        price=request.price,
    )
    db.add(new_property)
    db.commit()
    db.refresh(new_property)
    return new_property


@app.get(
    "/properties",
    response_model=list[schemas.PropertyOut],
    status_code=status.HTTP_200_OK,
)
def get_all_properties(db: DatabaseSession) -> list[models.Property]:
    """Retrieve all property records ordered deterministically by ID."""
    statement = select(models.Property).order_by(models.Property.id)
    return list(db.scalars(statement))


@app.get(
    "/properties/{property_id}",
    response_model=schemas.PropertyOut,
    status_code=status.HTTP_200_OK,
)
def get_property_by_id(property_id: int, db: DatabaseSession) -> models.Property:
    """Retrieve a single property record by its primary key ID or raise 404 Not Found."""
    db_property = db.get(models.Property, property_id)
    if not db_property:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Property with ID {property_id} was not found.",
        )
    return db_property


@app.put(
    "/properties/{property_id}",
    response_model=schemas.PropertyOut,
    status_code=status.HTTP_200_OK,
)
def update_property(
    property_id: int, request: schemas.PropertyUpdate, db: DatabaseSession
) -> models.Property:
    """Update an existing property record by its ID or raise 404 Not Found."""
    db_property = db.get(models.Property, property_id)
    if not db_property:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Property with ID {property_id} was not found.",
        )

    db_property.bedrooms = request.bedrooms
    db_property.bathrooms = request.bathrooms
    db_property.sqft_living = request.sqft_living
    db_property.sqft_lot = request.sqft_lot
    db_property.price = request.price

    db.commit()
    db.refresh(db_property)
    return db_property


@app.delete("/properties/{property_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_property(property_id: int, db: DatabaseSession):
    """Delete a property record from the database or raise 404 Not Found."""
    db_property = db.get(models.Property, property_id)
    if not db_property:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Property with ID {property_id} was not found.",
        )

    db.delete(db_property)
    db.commit()
    return None
