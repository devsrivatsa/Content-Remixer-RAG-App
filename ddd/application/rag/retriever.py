import concurrent.futures
import opik
from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue

from ddd.application import utils
from ddd.application.preprocessing.dispatcher import EmbeddingDispatcher
from ddd.domain.embedded_chunks import (
    EmbeddingChunk,
    EmbeddingChunkRepository,
    EmbeddingChunkPost,
    EmbeddingChunkArticle,
)
from ddd.domain.queries import Query, EmbeddedQuery
from .query_expansion import QueryExpansion
from .reranker import Reranker
from .self_query import SelfQuery

class ContextRetriever:
    def __init__(self, mock:bool=False) -> None:
        self._query_expander = QueryExpansion(mock=mock)
        self._metadata_extractor = SelfQuery(mock=mock)
        self._reranker = Reranker(mock=mock)
    
    @opik.track(name="ContextRetriever.search")
    def search(self, query:str, k:int=3, expand_to_n_queries:int=2) -> list:
        query_model = Query.from_str(query)
        query_model = self._metadata_extractor.generate(query_model)
        logger.info(f"successfully extracted the author_full_name = {query_model.author_full_name} from the query")

        n_generated_queries = self._query_expander.generate(query_model, expand_to_n=expand_to_n_queries)
        logger.info(f"Successfully generated {n_generated_queries} queries")

        with concurrent.futures.ThreadPoolExecutor() as exec:
            search_tasks = [exec.submit(self._search, _query_model, k) for _query_model in n_generated_queries]
            n_k_documents = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
            n_k_documents = utils.misc.flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))
        logger.info(f"{len(n_k_documents)} documents retrieced successfully")

        if len(n_k_documents) > 0:
            k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
        else:
            k_documents = []
        
        return k_documents

    def _search(self, query:Query, k:int=3) -> list[EmbeddingChunk]:
        assert k >= 3, "k should be >= 3"
        
        def _search_data_category(data_cat_odm: type[EmbeddingChunk], embedded_query: EmbeddedQuery) -> list[EmbeddingChunk]:
            if embedded_query.author_id:
                query_filter = Filter(must=[FieldCondition(key="author_id", match=MatchValue(value=str(embedded_query.author_id)))])
            else:
                query_filter = None
            
            return data_cat_odm.search(query_vector=embedded_query.embedding, limit=k//3, query_filter=query_filter)
        
        embedded_query: EmbeddedQuery = EmbeddingDispatcher.dispatch(query)
        
        post_chunks = _search_data_category(EmbeddingChunkPost, embedded_query)
        article_chunks = _search_data_category(EmbeddingChunkArticle, embedded_query)
        repositories_chunks = _search_data_category(EmbeddingChunkRepository, embedded_query)
        retrieved_chunks = post_chunks + article_chunks + repositories_chunks

        return retrieved_chunks
    
    def rerank(self, query:str|Query, chunks:list[EmbeddingChunk], keep_top_k:int) -> list[EmbeddingChunk]:
        if isinstance(query, str):
            query = Query.from_str(query)
        
        reranked_documents = self._reranker.generate(query=query, chunks=chunks, keep_top_k=keep_top_k)
        logger.info(f"{len(reranked_documents)} documents reranked successfully")
        
        return reranked_documents
