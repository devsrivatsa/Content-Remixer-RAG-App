from langchain.prompts import PromptTemplate
from .base import PromptTemplateFactory

class QueryExpansionTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI language model assistant. Your task is to generate {expand_to_n} different versions of the given user question to
    retrieve documents from a vector database. By generating multiple perspectives on the user question, your goal is help the user overcome some of 
    the limitations of the distance based similarity search.
    Provide these alternative questions separated by a '{separator}'.
    Original question: {question}"""

    @property
    def separator(self) -> str:
        return "#next-question#"
    
    def create_template(self, expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.separator,
                "expand_to_n": expand_to_n
            }
        )

class SelfQueryTemplate(PromptTemplateFactory):
    pronpt: str = """You are an AI language model assistant. Your task is to extract information from a user question.
    The required information that needs to be extracted is the user name or user id.
    Your response should consist of only the expected user name (e.g., John Smith) or id (e.g. 1324567), nothing else.
    If the user question does not contain any user name or id, return the following token: none.
    
    For example:
    QUESTION 1:
    My name is Srivatsa Sync and I want a post about...
    RESPONSE 1:
    Srivatsa Sync
    
    QUESTION 2:
    I want a post about...
    RESPONSE 2:
    none
    
    QUESTION 3:
    My user id is 1234567, and I want to..
    RESPONSE 3:
    1234567
    
    User question: {question}"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])