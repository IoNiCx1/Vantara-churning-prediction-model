"""
Database layer.

PostgreSQL is the specified store; DATABASE_URL defaults to a local
SQLite file so the API runs without a Postgres instance during
development.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./predictions.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    """Persisted churn/CLV scores."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, index=True, nullable=False)
    churn_probability = Column(Float, nullable=False)
    predicted_clv = Column(Float, nullable=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    scored_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SegmentAssignment(Base):
    """Persisted cluster/segment assignments."""

    __tablename__ = "segments"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, index=True, nullable=False)
    cluster_id = Column(Integer, nullable=False)
    business_label = Column(String, nullable=True)
    algorithm = Column(String, nullable=False)  # "kmeans" | "dbscan"
    assigned_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()