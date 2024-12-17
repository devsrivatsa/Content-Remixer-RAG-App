from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from typing_extensions import Annotated
from zenml import step, get_step_context

from ddd.application import utils
from ddd.domain.base.nosql import NoSQLBaseDocument
from ddd.domain.documents import ArticleDocument, Document, PostDocument, RepositoryDocument, UserDocument

@step
def query_data_warehouse(author_full_names: Annotated[list, "raw_documents"]) -> Annotated[list, "raw_documents"]:
    documents = []
    authors = []
    for author_full_name in author_full_names:
        logger.info(f"Quering data warehouse for {author_full_name}")        
        first_name, last_name = utils.split_full_name(author_full_name)
        logger.info(f"First name: {first_name}, Last name: {last_name}")
        user = UserDocument.get_or_create(first_name, last_name)
        authors.append(user)
        results = fetch_all_data(user)
        user_documents = [doc for query_result in results.values() for doc in query_result]
        documents.extend(user_documents)
    
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="raw_documents", metadata=_get_metadata(documents))

    return documents

def fetch_all_data(user: UserDocument) -> dict[str, list[NoSQLBaseDocument]]:
    user_id = str(user_id)
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(_fetch_articles, user_id): "articles",
            executor.submit(_fetch_posts, user_id): "posts", 
            executor.submit(_fetch_repositories, user_id): "repositories",
        }
        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed")
                results[query_name] = []
    
    return results

def _fetch_articles(user_id: str) -> list[NoSQLBaseDocument]:
    return ArticleDocument.bulk_find(author_id=user_id)

def _fetch_posts(user_id: str) -> list[NoSQLBaseDocument]:
    return PostDocument.bulk_find(author_id=user_id)

def _fetch_repositories(user_id: str) -> list[NoSQLBaseDocument]:
    return RepositoryDocument.bulk_find(author_id=user_id)

def _get_metadata(documents: list[Document]) -> dict:
    metadata = {
        "num_documents": len(documents),
    }
    for document in documents:
        collection = document.get_collection_name()
        if collection not in metadata:
            metadata[collection] = {}
        if "authors" not in metadata[collection]:
            metadata[collection]["authors"] = list()
        
        metadata[collection]["authors"].append(document.author_full_name)
    
    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))
    
    return metadata