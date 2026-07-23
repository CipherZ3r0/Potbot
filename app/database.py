"""
Database Repository — Abstract interface and SQLAlchemy/PostgreSQL implementation
following the Repository pattern.
"""

from abc import ABC, abstractmethod
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

import config
from domain.models import RAGResponse, FeedbackRecord

logger = logging.getLogger(__name__)

Base = declarative_base()


class ConversationEntity(Base):
    """SQLAlchemy ORM entity for conversations."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    context = Column(Text)
    model = Column(String(100))
    prompt_style = Column(String(50))
    retrieval_method = Column(String(50))
    response_time_ms = Column(Integer)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    reranking_used = Column(Boolean, default=False)
    query_rewriting_used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class FeedbackEntity(Base):
    """SQLAlchemy ORM entity for user feedback."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    sentiment = Column(String(20), nullable=False)
    comment = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class BaseDatabaseRepository(ABC):
    """Abstract Repository interface for persistent storage."""

    @abstractmethod
    def init_db(self) -> None:
        pass

    @abstractmethod
    def save_conversation(self, rag_response: RAGResponse, reranking_used: bool = False, query_rewriting_used: bool = False) -> int:
        pass

    @abstractmethod
    def save_feedback(self, feedback: FeedbackRecord) -> int:
        pass

    @abstractmethod
    def get_recent_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        pass


class PostgresDatabaseRepository(BaseDatabaseRepository):
    """PostgreSQL / SQLAlchemy implementation of Repository pattern."""

    def __init__(self, db_url: str = None):
        self.db_url = db_url or config.DATABASE_URL
        self._engine = None
        self._SessionLocal = None

    def _get_engine(self):
        if self._engine is None:
            self._engine = create_engine(self.db_url, pool_pre_ping=True)
        return self._engine

    def _get_session(self) -> Session:
        if self._SessionLocal is None:
            self._SessionLocal = sessionmaker(bind=self._get_engine())
        return self._SessionLocal()

    def init_db(self) -> None:
        engine = self._get_engine()
        Base.metadata.create_all(engine)
        logger.info("PostgresDatabaseRepository initialized database tables")

    def save_conversation(self, rag_response: RAGResponse, reranking_used: bool = False, query_rewriting_used: bool = False) -> int:
        session = self._get_session()
        try:
            sources_json = json.dumps([
                {
                    "file_name": doc.file_name,
                    "page_number": doc.page_number,
                    "text": doc.text[:200],
                }
                for doc in rag_response.retrieved_docs
            ])

            entity = ConversationEntity(
                question=rag_response.query,
                answer=rag_response.answer,
                context=sources_json,
                model=rag_response.model,
                prompt_style=rag_response.prompt_style,
                retrieval_method=rag_response.retrieval_method,
                response_time_ms=rag_response.response_time_ms,
                prompt_tokens=rag_response.prompt_tokens,
                completion_tokens=rag_response.completion_tokens,
                total_tokens=rag_response.total_tokens,
                reranking_used=reranking_used,
                query_rewriting_used=query_rewriting_used,
            )
            session.add(entity)
            session.commit()
            logger.info(f"Saved conversation record id={entity.id}")
            return entity.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving conversation: {e}")
            raise
        finally:
            session.close()

    def save_feedback(self, feedback: FeedbackRecord) -> int:
        session = self._get_session()
        try:
            entity = FeedbackEntity(
                conversation_id=feedback.conversation_id,
                sentiment=feedback.sentiment,
                comment=feedback.comment,
            )
            session.add(entity)
            session.commit()
            logger.info(f"Saved feedback id={entity.id} for conversation_id={feedback.conversation_id}")
            return entity.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving feedback: {e}")
            raise
        finally:
            session.close()

    def get_recent_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        session = self._get_session()
        try:
            rows = (
                session.query(ConversationEntity)
                .order_by(ConversationEntity.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "question": r.question,
                    "answer": r.answer,
                    "model": r.model,
                    "response_time_ms": r.response_time_ms,
                    "total_tokens": r.total_tokens,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
        finally:
            session.close()
