"""
RhythmAI — Database Layer
SQLite via SQLAlchemy (swap to PostgreSQL by changing DATABASE_URL)
Tables: users, analyses, analysis_scores, audit_log
"""

import os
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, LargeBinary
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
import enum

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR}/rhythmai.db")

engine       = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class UserRole(str, enum.Enum):
    doctor   = "doctor"
    resident = "resident"
    student  = "student"
    admin    = "admin"

class UserStatus(str, enum.Enum):
    pending   = "pending"
    verified  = "verified"
    suspended = "suspended"


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String(120), nullable=False)
    email           = Column(String(200), unique=True, index=True, nullable=False)
    password_hash   = Column(String(200), nullable=False)
    role            = Column(String(20), default=UserRole.doctor)
    medical_id      = Column(String(80), unique=True, nullable=False)
    specialty       = Column(String(120), nullable=True)
    hospital        = Column(String(200), nullable=True)
    status          = Column(String(20), default=UserStatus.pending)
    joined_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_login      = Column(DateTime, nullable=True)
    is_admin        = Column(Boolean, default=False)
    # Doctor ID card photo (stored as base64 string)
    id_card_photo   = Column(Text, nullable=True)
    id_card_filename= Column(String(200), nullable=True)

    analyses        = relationship("Analysis", back_populates="doctor", cascade="all, delete-orphan")
    audit_logs      = relationship("AuditLog", back_populates="user",   cascade="all, delete-orphan")


class Analysis(Base):
    __tablename__ = "analyses"

    id              = Column(Integer, primary_key=True, index=True)
    ecg_id          = Column(String(20), unique=True, index=True)
    doctor_id       = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_ref     = Column(String(80), nullable=True)
    patient_age     = Column(Integer, nullable=True)
    patient_sex     = Column(String(2), nullable=True)
    input_mode      = Column(String(10), nullable=False)
    result          = Column(String(6), nullable=False)
    confidence      = Column(Float, nullable=False)
    inference_time  = Column(Float, nullable=True)
    notes           = Column(Text, default="")
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                             onupdate=lambda: datetime.now(timezone.utc))

    doctor          = relationship("User", back_populates="analyses")
    scores          = relationship("AnalysisScore", back_populates="analysis",
                                   cascade="all, delete-orphan")


class AnalysisScore(Base):
    __tablename__ = "analysis_scores"

    id          = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    label       = Column(String(6), nullable=False)
    probability = Column(Float, nullable=False)

    analysis    = relationship("Analysis", back_populates="scores")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=True)
    action      = Column(String(80), nullable=False)
    target      = Column(String(120), nullable=True)
    detail      = Column(Text, nullable=True)
    ip_address  = Column(String(50), nullable=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user        = relationship("User", back_populates="audit_logs")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)


def next_ecg_id(db: Session) -> str:
    count    = db.query(Analysis).count()
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"ECG-{date_str}-{str(count + 1).zfill(4)}"
