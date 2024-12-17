from loguru import logger
from ddd.domain.base import NoSQLBaseDocument, VectorBaseDocument
from ddd.domain.types import DataCategory

from .cleaning_data_handlers import (
    CleaningDataHandler,
    PostCleaningDataHandler,
    ArticleCleaningDataHandler,
    RepositoryCleaningDataHandler,
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




