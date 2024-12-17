from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from ddd.settings import settings

class MongoDatabaseConnector:
    _instance: MongoClient | None = None

    def __new__(cls, *args, **kwargs) -> MongoClient:
        if cls._instance is None:
            try:
                cls._instance = MongoClient(settings.DATABASE_HOST)
            except ConnectionFailure as err:
                logger.error(f"Couldn't connect to the mongodb database: {err!s}")
                raise
        logger.info(f"Connected to the mongodb database: {settings.DATABASE_HOST}")
        
        return cls._instance

connection = MongoDatabaseConnector()