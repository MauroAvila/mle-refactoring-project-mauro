"""Integration test suite for the King County REST API.

This module verifies the integration between FastAPI HTTP endpoints (service/main.py),
Pydantic schema validation, and database persistence using TestClient and an
in-memory SQLite database.
"""

import os

# Set dummy DB_CONN env variable BEFORE importing application modules
os.environ["DB_CONN"] = "sqlite:///:memory:"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from service.database import Base, get_db
from service.main import app

# In-memory SQLite database engine for complete test isolation
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture(name="db_session")
def override_get_db():
    """Fixture to create tables in memory, yield a clean session, and teardown afterwards."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(name="client")
def client_fixture(db_session: Session):
    """Fixture to override API dependency injection with the ephemeral database session."""
    def get_db_override():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = get_db_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# --- HTTP ROUTE INTEGRATION TESTS ---

def test_root_health_check(client: TestClient):
    """Verify that the root endpoint returns HTTP status 200 and 'ok' status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_property_integration(client: TestClient):
    """Verify property creation via POST /properties returning HTTP 201 Created."""
    payload = {
        "bedrooms": 3,
        "bathrooms": 2.5,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "price": 450000.0,
    }
    response = client.post("/properties", json=payload)
    assert response.status_code in (200, 201)

    data = response.json()
    assert data["bedrooms"] == 3
    assert data["price"] == 450000.0


def test_get_all_properties_integration(client: TestClient):
    """Verify fetching the list of all properties via GET /properties."""
    payload = {
        "bedrooms": 4,
        "bathrooms": 3.0,
        "sqft_living": 2400,
        "sqft_lot": 6000,
        "price": 600000.0,
    }
    client.post("/properties", json=payload)

    response = client.get("/properties")
    assert response.status_code == 200
    properties = response.json()

    assert isinstance(properties, list)
    assert len(properties) >= 1


def test_get_property_by_id_integration(client: TestClient):
    """Verify fetching a single property by ID and handling non-existent ID (404 Not Found)."""
    payload = {
        "bedrooms": 2,
        "bathrooms": 1.0,
        "sqft_living": 1000,
        "sqft_lot": 3000,
        "price": 250000.0,
    }
    create_res = client.post("/properties", json=payload)
    item_id = create_res.json().get("id", 1)

    # Successful fetch
    get_res = client.get(f"/properties/{item_id}")
    assert get_res.status_code == 200
    assert get_res.json()["bedrooms"] == 2

    # Fetch non-existent record
    missing_res = client.get("/properties/999999")
    assert missing_res.status_code == 404


def test_update_property_integration(client: TestClient):
    """Verify updating an existing property record via PUT /properties/{id}."""
    initial_payload = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1500,
        "sqft_lot": 4000,
        "price": 350000.0,
    }
    create_res = client.post("/properties", json=initial_payload)
    item_id = create_res.json().get("id", 1)

    update_payload = {
        "bedrooms": 4,
        "bathrooms": 2.5,
        "sqft_living": 1500,
        "sqft_lot": 4000,
        "price": 400000.0,
    }
    put_res = client.put(f"/properties/{item_id}", json=update_payload)
    assert put_res.status_code == 200
    assert put_res.json()["bedrooms"] == 4
    assert put_res.json()["price"] == 400000.0


def test_delete_property_integration(client: TestClient):
    """Verify property deletion via DELETE /properties/{id}."""
    payload = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1200,
        "sqft_lot": 3500,
        "price": 300000.0,
    }
    create_res = client.post("/properties", json=payload)
    item_id = create_res.json().get("id", 1)

    # Delete property
    del_res = client.delete(f"/properties/{item_id}")
    assert del_res.status_code in (200, 204)

    # Verify deleted property no longer exists
    get_res = client.get(f"/properties/{item_id}")
    assert get_res.status_code == 404