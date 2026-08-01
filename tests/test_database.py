"""Unit tests for PostgresDatabaseRepository (using in-memory SQLite for testing)."""

import pytest
from app.database import PostgresDatabaseRepository
from domain.models import RAGResponse, FeedbackRecord, SearchResult


@pytest.fixture
def repo(tmp_path):
    # Use SQLite memory DB for testing
    db_url = "sqlite:///:memory:"
    repo = PostgresDatabaseRepository(db_url=db_url)
    repo.init_db()
    return repo


def test_save_conversation(repo, sample_search_results):
    rag_resp = RAGResponse(
        answer="A", query="Q", rewritten_query=None,
        retrieved_docs=sample_search_results,
        model="model", prompt_style="concise",
        retrieval_method="hybrid", response_time_ms=100
    )
    
    conv_id = repo.save_conversation(rag_resp)
    assert conv_id == 1
    
    recent = repo.get_recent_conversations(limit=10)
    assert len(recent) == 1
    assert recent[0]["question"] == "Q"


def test_save_feedback(repo, sample_search_results):
    rag_resp = RAGResponse(
        answer="A", query="Q", rewritten_query=None,
        retrieved_docs=[], model="m", prompt_style="s",
        retrieval_method="m", response_time_ms=100
    )
    conv_id = repo.save_conversation(rag_resp)
    
    feedback = FeedbackRecord(conversation_id=conv_id, sentiment="positive", comment="Good")
    fb_id = repo.save_feedback(feedback)
    assert fb_id == 1


def test_get_recent_conversations_limit(repo):
    rag_resp = RAGResponse(
        answer="A", query="Q", rewritten_query=None,
        retrieved_docs=[], model="m", prompt_style="s",
        retrieval_method="m", response_time_ms=100
    )
    for _ in range(5):
        repo.save_conversation(rag_resp)
        
    recent = repo.get_recent_conversations(limit=3)
    assert len(recent) == 3
