import uuid
from abc import ABC
from typing import Generic, TypeVar, Type
from loguru import logger
from pydantic import BaseModel, UUID4, Field
from pymongo import errors

from ddd.domain.exceptions import ImproperlyConfigured
from ddd.infrastructure.db.mongo import connection
from ddd.settings import settings

_database = connection.get_database(settings.DATABASE_NAME)

T = TypeVar("T", bound="NoSQLBaseDocument")

class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False
        return self.id == value.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    @classmethod
    def from_mongo(cls: Type[T], data: dict) -> T:
        """convert a mongo document to a pydantic model"""
        if not data:
            raise ValueError("Data is empty")
        id = data.pop("_id")
        
        return cls(**dict(id=id, **data))

    def to_mongo(self: T, **kwargs) -> dict:
        """convert a pydantic model to a mongo document"""
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)
        parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)
        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))
        for k, v in parsed.items():
            if isinstance(v, uuid.UUID):
                parsed[k] = str(v)
        
        return parsed
    
    def model_dump(self:T, **kwargs) -> dict:
        dict_ = super().model_dump(**kwargs)
        for k, v in dict_.items():
            if isinstance(v, uuid.UUID):
                dict_[k] = str(v)
        return dict_
    
    def save(self:T, **kwargs) -> T | None:
        collection = _database[self.get_collection_name()]
        try:
            collection.insert_one(self.to_mongo(**kwargs))
            
            return self
        except errors.WriteError:
            logger.exception("Failed to insert document")

            return None
    
    
    @classmethod
    def get_or_create(cls: Type[T], **filter_options) -> T:
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)
            
            new_instance = cls(**filter_options)
            new_instance = new_instance.save()
            return new_instance
        except errors.OperationFailure:
            logger.error(f"Failed to get documents with the given filter options: {filter_options}")
            raise
    
        
    
    @classmethod
    def find(cls: Type[T], **filter_options) -> T | None:
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)
            
            return None
        except errors.OperationFailure:
            logger.error("Failed to retrieve document")

            return None
    
    @classmethod
    def bulk_find(cls: Type[T], **filter_options) -> list[T]:
        collection = _database[cls.get_collction_name()]
        try:
            instances = collection.find(filter_options)
        
            return [doc for instance in instances if (doc := cls.from_mongo(instance)) is not None]
        except errors.OperationFailure:
            logger.error("Failed to retrieve documents")

            return []


    @classmethod
    def bulk_insert(cls: Type[T], documents: list[T], **kwargs) -> bool:
        collection = _database[cls.get_collection_name()]
        try:
            collection.insert_many(doc.to_mongo(**kwargs) for doc in documents)

            return True
        except (errors.WriteError, errors.BulkWriteError):
            logger.error(f"Failed to insert documents of type {cls.__name__}")

            return False
    
    @classmethod
    def get_collection_name(cls: Type[T]) -> str:
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise ImproperlyConfigured("Document class should define a Settings configuration class with the name of the collection")
        
        return cls.Settings.name
