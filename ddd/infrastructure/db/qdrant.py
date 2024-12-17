from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from ddd.settings import settings

class QdrantDatabaseConnector:
    _instance: QdrantClient | None = None

    def __new__(cls, *args, **kwargs) -> QdrantClient:
        if cls._instance is None:
            try:
                if settings.USE_QDRANT_CLOUD:
                    cls._instance = QdrantClient(
                        url=settings.QDRANT,
                        api_key=settings.QDRANT_API_KEY
                    )
                    uri = settings.QDRANT_CLOUD_URI
                else:
                    cls._instance = QdrantClient(
                        host=settings.QDRANT_DATABASE_HOST,
                        port=settings.QDRANT_DATABASE_PORT,
                    )
                    url = f"{settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}"
                
                logger.info(f"Connection to Qdrant DB with URI successful: {uri}")
            
            except UnexpectedResponse:
                logger.exception(
                    "Couldn't connect to Qdrant DB",
                    host = settings.QDRANT_DATABASE_HOST,
                    port = settings.QDRANT_DATABASE_PORT,
                    url = settings.QDRANT_CLOUD_URI,
                )
                raise
            
            return cls._instance

connection = QdrantDatabaseConnector()