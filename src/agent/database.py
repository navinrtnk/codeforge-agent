"""Database engine and session management."""

from collections.abc import Generator

from fastapi import Request
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from agent.models import Base


class Database:
    """Own the SQLAlchemy engine and session factory."""

    def __init__(self, url: str) -> None:
        engine_options: dict[str, object] = {}
        if url.startswith("sqlite"):
            engine_options["connect_args"] = {"check_same_thread": False}
        if url in {"sqlite://", "sqlite:///:memory:"}:
            engine_options["poolclass"] = StaticPool

        self.engine: Engine = create_engine(url, **engine_options)
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )

    def create_schema(self) -> None:
        """Create tables that do not already exist."""
        Base.metadata.create_all(self.engine)

    def session(self) -> Generator[Session]:
        """Yield a database session and always close it afterward."""
        with self.session_factory() as session:
            yield session

    def dispose(self) -> None:
        """Release pooled database connections."""
        self.engine.dispose()


def get_database_session(request: Request) -> Generator[Session]:
    """Provide a session from the database attached to the FastAPI app."""
    database: Database = request.app.state.database
    yield from database.session()
