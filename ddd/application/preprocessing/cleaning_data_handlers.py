from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ddd.domain.cleaned_documents import (
    CleanedDocument,
    CleanedPostDocument,
    CleanedArticleDocument,
    CleanedRepositoryDocument,
)
from ddd.domain.documents import (
    Document,
    PostDocument,
    RepositoryDocument,
    ArticleDocument
)

from .operations import clean_text

DocumentT = TypeVar("DocumentT", bound=Document)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)

class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
    """Abstract base class for cleaning data handlers.
    Contains data cleaning logic for a specific type of document."""

    @abstractmethod
    def clean(self, data_model:DocumentT) -> CleanedDocumentT:
        pass

class PostCleaningDataHandler(CleaningDataHandler):
    def clean(self, data_model: PostDocument) -> CleanedPostDocument:
        return CleanedPostDocument(
            id = data_model.id,
            content = clean_text(" ### ".join(data_model.content.values())),
            platform = data_model.platform,
            author_id = data_model.author_id,
            author_full_name = data_model.author_full_name,
            image = data_model.image if data_model.image else None
        )

class ArticleCleaningDataHandler(CleaningDataHandler):
    def clean(self, data_model: ArticleDocument) -> CleanedArticleDocument:
        valid_content = [content for content in data_model.content.values() if content]
        return CleanedArticleDocument(
            id = data_model.id,
            content = clean_text(" ### ".join(valid_content)),
            platform = data_model.platform,
            author_id = data_model.author_id,
            author_full_name = data_model.author_full_name,
            link = data_model.link,
        )

class RepositoryCleaningDataHandler(CleaningDataHandler):
    def clean(self, data_model: RepositoryDocument) -> CleanedRepositoryDocument:
        return CleanedRepositoryDocument(
            id = data_model.id,
            content = clean_text(" ### ".join(data_model.content.values())),
            platform = data_model.platform,
            author_id = data_model.author_id,
            author_full_name = data_model.author_full_name,
            link = data_model.link,
            name = data_model.name,
        )