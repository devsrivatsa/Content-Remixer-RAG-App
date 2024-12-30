from loguru import logger
from ddd.domain.base import NoSQLBaseDocument, VectorBaseDocument
from ddd.domain.types import DataCategory

from .cleaning_data_handlers import (
    CleaningDataHandler,
    PostCleaningDataHandler,
    ArticleCleaningDataHandler,
    RepositoryCleaningDataHandler,
)

from .chunking_data_handlers import (
    ChunkingDataHandler,
    PostChunkingHandler,
    ArticleChunkingHandler,
    RepositoryChunkingHandler,
)

from .embedding_data_handlers import (
    ArticleEmbeddingHandler,
    EmbeddingDataHandler,
    PostEmbeddingHandler,
    RepositoryEmbeddingHandler,
    QueryEmbeddingHandler,
)

class CleaningHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> CleaningDataHandler:
        if data_category == DataCategory.POSTS:
            return PostCleaningDataHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleCleaningDataHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryCleaningDataHandler()
        else:
            raise ValueError(f"No cleaning handler found for data category: {data_category}")

class CleaningDispatcher:
    cleaning_factory = CleaningHandlerFactory()

    @classmethod
    def dispatch(cls, data_model: NoSQLBaseDocument) -> VectorBaseDocument:
        data_category = DataCategory(data_model.get_collection_name())
        handler = cls.cleaning_factory.create_handler(data_category)
        clean_model = handler.clean(data_model)
        logger.info(
            "Document cleaned successfully",
            data_category=data_category,
            cleaned_content_len = len(clean_model.content),
        )
        return clean_model
    
class ChunkingHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> ChunkingDataHandler:
        if data_category == DataCategory.POSTS:
            return PostChunkingHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleChunkingHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryChunkingHandler()
        else:
            raise ValueError(f"No chunking handler found for data category: {data_category}")
        
class ChunkingDispatcher:
    chunking_factory = ChunkingHandlerFactory

    @classmethod
    def dispatch(cls, data_model: VectorBaseDocument) -> list[VectorBaseDocument]:
        data_category = data_model.get_category()
        handler = cls.chunking_factory.create_handler(data_category)
        chunked_models = handler.chunk(data_model)

        logger.info("Document chunked successfully", data_category=data_category, num=len(chunked_models))

        return chunked_models

class EmbeddingHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> EmbeddingDataHandler:
        if data_category == DataCategory.POSTS:
            return PostEmbeddingHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleEmbeddingHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryEmbeddingHandler()
        elif data_category == DataCategory.QUERIES:
            return QueryEmbeddingHandler()
        else:
            raise ValueError(f"No embedding handler found for data category: {data_category}")

class EmbeddingDispatcher:
    embedding_factory = EmbeddingHandlerFactory
    
    @classmethod
    def diapatch(cls, data_model:VectorBaseDocument | list[VectorBaseDocument]) -> VectorBaseDocument | list[VectorBaseDocument]:
        is_list = isinstance(data_model, list)
        if not is_list:
            data_model = [data_model]
        if len(data_model) == 0:
            return []
        data_category = data_model[0].get_category()
        
        assert all(
            dm.get_category() == data_category
            for dm in data_model
        ), "All documents must be of the same category"

        handler = cls.embedding_factory.create_handler(data_category)
        embedded_chunk_model = handler.embed_batch(data_model)
        if not is_list:
            embed_chunk_model = embed_chunk_model[0]
        
        logger.info("Data embedded successfully", data_category=data_category)

        return embedded_chunk_model
    
