import opik
from langchain_openai import ChatOpenAI
from loguru import logger

from ddd.domain.queries import Query
from ddd.settings import settings

from .base import RAGStep
from .prompt_templates import QueryExpansionTemplate

class QueryExpansion(RAGStep):
    @opik.track(name="QueryExpansion.generate")
    def generate(self, query: Query, expand_to_n: int) -> list[str]:
        assert expand_to_n > 0, f"'expand_to_n' should be greater than 0. Got {expand_to_n}"

        if self._mock:
            return [query for _ in range(expand_to_n)]
        
        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(expand_to_n-1)
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY, temperature=0)
        chain = prompt | model
        response = chain.invoke({"question": query})
        result = response.content
        queries_content = result.strip().split(query_expansion_template.separator)
        queries = [query]
        queries += [
            query.replace_content(striped_content)
            for content in queries_content
            if (striped_content := content.strip())
        ]
        
        return queries

if __name__ == "__main__":
    query = Query.from_str("write an article about advanced RAG methods.")
    query_expander = QueryExpansion()
    expanded_queries = query_expander.generate(query, expand_to_n=3)
    for expanded_query in expanded_queries:
        logger.info(expanded_query.content)
    