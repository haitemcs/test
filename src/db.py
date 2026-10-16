"""SQLAlchemy engine, ORM table definitions, and database initialization."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config import settings

# The engine manages database connections. pool_pre_ping checks a connection before
# using it, helping detect connections that have gone stale.
engine = create_engine(settings.database_url, pool_pre_ping=True)

# SessionLocal is a factory. Calling SessionLocal() creates a session for one unit
# of database work; the session is not itself the database.
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Parent class used by SQLAlchemy to collect table definitions."""

    pass


class Experiment(Base):
    """Model evaluation summary; currently training writes metrics to JSON instead."""

    __tablename__ = "experiments"

    # Each mapped attribute corresponds to a column in the SQL table.
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100))
    mae: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    r2: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """One prediction returned by the API, plus a small record of its input."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    prediction: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(100))
    request_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def init_db():
    """Create tables that do not exist yet.

    create_all is convenient in a learning project, but it does not update an existing
    table when a model changes. Use database migrations as the schema evolves.
    """
    Base.metadata.create_all(bind=engine)
