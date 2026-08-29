"""
RAG Pipeline Orchestrator — Clean Facade & Dependency Injection architecture.
"""

import logging
from typing import Optional

import config
from domain.models import RAGResponse
from rag.retrievers import BaseSearchStrategy, HybridSearchStrategy, SearchStrategyFactory
from rag.rerankers import BaseReranker, CrossEncoderReranker, NoOpReranker
from rag.query_rewriters import BaseQueryRewriter, LLMQueryRewriter, NoOpQueryRewriter
from rag.prompt_builders import BasePromptBuilder, TemplatePromptBuilder
from rag.llm_providers import BaseLLMProvider, GroqLLMProvider, create_llm_provider
from app.database import BaseDatabaseRepository, PostgresDatabaseRepository

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Facade orchestrating Query Rewriting, Search Retrieval, Document Re-ranking, Prompt Construction, and LLM Generation."""

    def __init__(
        self,
        search_strategy: Optional[BaseSearchStrategy] = None,
        reranker: Optional[BaseReranker] = None,
        query_rewriter: Optional[BaseQueryRewriter] = None,
        prompt_builder: Optional[BasePromptBuilder] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        repository: Optional[BaseDatabaseRepository] = None,
    ):
        self.search_strategy = search_strategy or HybridSearchStrategy()
        self.reranker = reranker or CrossEncoderReranker()
        self.query_rewriter = query_rewriter or LLMQueryRewriter()
        self.prompt_builder = prompt_builder or TemplatePromptBuilder()
        self.llm_provider = llm_provider or create_llm_provider()
        self.repository = repository or PostgresDatabaseRepository()

    def query(
        self,
        user_query: str,
        retrieval_method: str = "hybrid",
        use_reranking: bool = True,
        use_query_rewriting: bool = True,
        prompt_style: str = "detailed",
        llm_provider_name: str = None,
        llm_model: str = None,
        top_k: int = 5,
        rerank_top_n: int = 3,
        save_to_db: bool = True,
    ) -> RAGResponse:
        """Execute complete RAG flow for a user query."""

        # 1. Select search strategy
        if retrieval_method != "hybrid":
            strategy = SearchStrategyFactory.get_strategy(retrieval_method)
        else:
            strategy = self.search_strategy

        # Get the active LLM provider for this query
        active_llm_provider = create_llm_provider(llm_provider_name) if llm_provider_name else self.llm_provider

        # 2. Query Rewriting (optional)
        if use_query_rewriting:
            rewriter = LLMQueryRewriter(llm_provider=active_llm_provider, model=llm_model)
        else:
            rewriter = NoOpQueryRewriter()
        
        search_q = rewriter.rewrite(user_query)

        # 3. Retrieval
        fetch_k = top_k * 2 if use_reranking else top_k
        raw_results = strategy.search(search_q, top_k=fetch_k)

        # 4. Re-ranking (optional)
        reranker_svc = self.reranker if use_reranking else NoOpReranker()
        final_results = reranker_svc.rerank(user_query, raw_results, top_n=rerank_top_n)

        # 5. Prompt Construction
        messages = self.prompt_builder.build_prompt(user_query, final_results, style=prompt_style)

        # 6. LLM Generation
        gen_result = active_llm_provider.generate(messages, model=llm_model)

        # 7. Assemble RAGResponse domain object
        rag_response = RAGResponse(
            answer=gen_result["answer"],
            query=user_query,
            rewritten_query=search_q if use_query_rewriting else None,
            retrieved_docs=final_results,
            model=gen_result["model"],
            prompt_style=prompt_style,
            retrieval_method=retrieval_method,
            response_time_ms=gen_result["response_time_ms"],
            prompt_tokens=gen_result["prompt_tokens"],
            completion_tokens=gen_result["completion_tokens"],
            total_tokens=gen_result["total_tokens"],
        )

        # 8. Save interaction to Database Repository
        if save_to_db and self.repository:
            try:
                conv_id = self.repository.save_conversation(
                    rag_response,
                    reranking_used=use_reranking,
                    query_rewriting_used=use_query_rewriting,
                )
                rag_response.conversation_id = conv_id
            except Exception as e:
                logger.warning(f"Failed to persist conversation record: {e}")

        return rag_response
