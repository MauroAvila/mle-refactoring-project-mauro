# King County Housing — Refactoring & Production Pipeline

This repository contains the production-grade refactoring of the King County housing price prediction project. It transforms an exploratory, notebook-shaped workflow (`King-County.ipynb`) into a modular, testable, fully containerized Python application following MLOps and software engineering best practices.

---

## Project Overview

The primary goal of this project is to separate exploratory data analysis (EDA) from stable, deterministic production logic. The codebase is organized into reusable Python modules, an automated pipeline CLI, a machine learning training workflow, a RESTful API backed by a PostgreSQL database, a multi-container Docker infrastructure, and a comprehensive `pytest` testing suite.

---

## Repository Structure

* `src/king_county/`
  * `data_cleaning.py`: Pure, immutable functions for filtering outliers, rebuilding basement area, static NaN imputation, and temporal feature consolidation.
  * `feature_engineering.py`: Vectorized functions for price per square foot and geodetic distances to Medina (wealth center) and waterfront properties with Earth curvature correction.
  * `pipeline.py`: Command-line interface (CLI) script that orchestrates the end-to-end data cleaning and feature transformation workflow reproducibly.
  * `train.py`: Leak-free training module configuring a scikit-learn pipeline with `PolynomialFeatures(degree=2)`, `StandardScaler`, and `ElasticNet` wrapped in a `TransformedTargetRegressor`. Optimizes hyperparameters via `GridSearchCV` and exports the trained estimator using `skops`.
* `service/`
  * `database.py`: SQLAlchemy engine, session factory (`SessionLocal`), and request-scoped session generator (`get_db`).
  * `models.py`: SQLAlchemy ORM mapping for the physical `properties` table in PostgreSQL.
  * `schemas.py`: Pydantic v2 data models declaring strict validation rules for API request payloads (`PropertyCreate`, `PropertyUpdate`) and response models (`PropertyOut`).
  * `main.py`: FastAPI application implementing full HTTP CRUD endpoints (`POST`, `GET`, `PUT`, `DELETE`).
  * `Dockerfile`: Multi-stage Docker build recipe leveraging `uv` for fast, locked dependency installation.
* `tests/`
  * `unit/test_data_processing.py`: Unit test suite using `pytest` and `pandas.testing` (`assert_frame_equal`, `assert_series_equal`) over isolated synthetic DataFrames.
  * `integration/test_api.py`: Integration test suite validating FastAPI endpoints, Pydantic contracts, and database persistence using `TestClient` and an in-memory SQLite database (`sqlite:///:memory:`).
* `docker-compose.yaml`: Multi-container orchestration managing PostgreSQL 17 (`db`) with persistent volume storage and the FastAPI web service (`app`) with active healthchecks and dependency ordering.
* `.env`: Environment file declaring database credentials and connection strings (`DB_CONN`).
* `pyproject.toml`: Modern Python project specification and dependency declarations.

---

## Quick Start & Execution Guide

### 1. Prerequisites and Setup

Clone the repository and install dependencies using `uv`:

```bash
git clone <your-repository-url>
cd king_county_project
uv sync