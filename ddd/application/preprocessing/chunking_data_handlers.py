import hashlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from ddd.domain.chunks import ArticleChunk, Chunk, PostChunk, RepositoryChunk
from ddd.domain.cleaned_documents import (
    CleanedDocument,
    CleanedPostDocument,
    CleanedArticleDocument,
    CleanedRepositoryDocument,
)
from .operations import chunk_article, chunk_text

CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)
ChunkT = TypeVar("ChunkT", bound=Chunk)

class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]):
    """Abstract base class for chunking data handlers. Contains methods for chunking data."""

    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50
        }
    
    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> list[ChunkT]:
        pass

class PostChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "min_length": 250,
            "max_length": 25
        }
    
    def chunk(self, data_model: CleanedPostDocument) -> list[PostChunk]:
        data_models_list = []
        cleaned_content = data_model.content
        chunks = chunk_text(cleaned_content, chunk_size=self.metadata["chunk_size"], chunk_overlap=self.metadata["chunk_overlap"])
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = PostChunk(
                id = UUID(chunk_id, version=4),
                content = chunk,
                platform = data_model.platform,
                document_id = data_model.id,
                author_id = data_model.author_id,
                author_full_name = data_model.author_full_name,
                image = data_model.image if data_model.image else None,
                metadata = self.metadata
            )
            data_models_list.append(model)

        return data_models_list


class ArticleChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "min_length": 1000,
            "max_length": 2000
        }

    def chunk(self, data_model: CleanedArticleDocument) -> list[ArticleChunk]:
        data_models_list = []
        cleaned_content = data_model.content
        chunks = chunk_text(cleaned_content, chunk_size=self.metadata["chunk_size"], chunk_overlap=self.metadata["chunk_overlap"])
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = ArticleChunk(
                id = UUID(chunk_id, version=4),
                content = chunk,
                platform = data_model.platform,
                document_id = data_model.id,
                author_id = data_model.author_id,
                author_full_name = data_model.author_full_name,
                link = data_model.link,
                metadata = self.metadata
            )
            data_models_list.append(model)

        return data_models_list

class RepositoryChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "min_length": 1500,
            "max_length": 100
        }
    
    def chunk(self, data_model: CleanedRepositoryDocument) -> list[RepositoryChunk]:
        data_models_list = []
        cleaned_content = data_model.content
        chunks = chunk_text(cleaned_content, chunk_size=self.metadata["chunk_size"], chunk_overlap=self.metadata["chunk_overlap"])
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = RepositoryChunk(
                id = UUID(chunk_id, version=4),
                content = chunk,
                platform = data_model.platform,
                document_id = data_model.id,
                name = data_model.name,
                link = data_model.link,
                author_id = data_model.author_id,
                author_full_name = data_model.author_full_name,
                metadata = self.metadata
            )
            data_models_list.append(model)
        
        return data_models_list