from zenml import pipeline
from steps import feature_engineering as fe

@pipeline
def feature_engineering(author_full_names: list[str], wait_for: str|list[str]|None=None) -> list[str]:
    raw_documents = fe.query_data_warehouse(author_full_names, after=wait_for)
    cleaned_documents = fe.clean_documents(raw_documents)
    last_step1 = fe.load_to_vector_db(cleaned_documents)
    embedded_documents = fe.chunk_and_embed(cleaned_documents)
    last_step2 = fe.load_to_vector_db(embedded_documents)

    return [last_step1.invocation_id, last_step2.invocation_id]
