import opik
from langchain_openai import ChatOpenAI
from loguru import logger

from ddd.application import utils
from ddd.domain.documents import UserDocument
from ddd.domain.queries import Query
from ddd.settings import settings

from .base import RAGStep
from .prompt_templates import SelfQueryTemplate

class SelfQuery(RAGStep):
    @opik.track(name="SelfQuery.generate")
    def generate(self, query: Query) -> Query:
        if self._mock:
            return query
        
        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY, temperature=0)
        chain = prompt | model
        response = chain.invoke({"question": query})
        user_full_name = response.content.strip("\n")

        if user_full_name == "none":
            return query
        first_name, last_name = utils.split_user_full_name(user_full_name)
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        query.author_id = user.id
        query.author_full_name = user.full_name

        return query

if __name__ == "__main__":
    query = Query.from_str("I am Srivatsa Smith. Write an article about advanced RAG methods.")
    self_query = SelfQuery()
    query = self_query.generate(query)
    logger.info(f"Extracted author_id: {query.author_id}")
    logger.info(f"Extracted author_full_name: {query.author_full_name}")
