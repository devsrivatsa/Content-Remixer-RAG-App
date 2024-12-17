from abc import ABC, abstractmethod
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from ddd.domain.queries import Query
from typing import Any


class PromptTemplateFactory(ABC, BaseModel):

    @abstractmethod
    def create_template(self) -> PromptTemplate:
        pass

class RAGStep(ABC):
    def __init__(self, mock:bool=False) -> None:
        self._mock = mock

    @abstractmethod
    def generate(self, query:Query, *args, **kwargs) -> Any:
        pass
    