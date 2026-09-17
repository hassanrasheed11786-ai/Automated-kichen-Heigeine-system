import datetime
import os
from pathlib import Path
from typing import Generator

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_URL = f"sqlite:///{BASE_DIR / 'hygiene.db'}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

engine_kwargs = (
    {"connect_args": {"check_same_thread": False}}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class DetectionLog(Base):
    """Time-series logging for hygiene analytics."""
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)
    person_id = Column(String, index=True, nullable=False)
    class_name = Column(String, index=True, nullable=False)
    status = Column(String, index=True, nullable=False)


class HighlightClip(Base):
    """Generated violation/highlight clips."""
    __tablename__ = "highlight_clips"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)
    person_id = Column(String, index=True, nullable=False)
    reason = Column(String, nullable=False)
    clip_url = Column(String, nullable=False)


class HygieneEvent(Base):
    """A temporally-stable person/class observation from one analyzed video.

    Unlike DetectionLog this represents one incident interval, never one model
    detection per frame.  Violations optionally have a generated highlight.
    """
    __tablename__ = "hygiene_events"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)
    source_id = Column(String, index=True, nullable=False)
    person_id = Column(String, index=True, nullable=False)
    class_name = Column(String, index=True, nullable=False)
    status = Column(String, index=True, nullable=False)
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    start_seconds = Column(Float, nullable=False)
    end_seconds = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    clip_url = Column(String, nullable=True)


class ViolationLog(Base):
    """Violation snapshots created by the detector pipeline."""
    __tablename__ = "violation_logs"

    id = Column(Integer, primary_key=True, index=True)
    worker_id = Column(String, index=True, nullable=False)
    violation_type = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)
    snapshot_path = Column(String, nullable=False)


def init_db() -> None:
    """Create all missing database tables automatically."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI database dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Automatically create missing tables.
init_db()
