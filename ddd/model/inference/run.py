from __future__ import annotations
from ddd.domain.inference import Inference
from ddd.settings import settings

class InferenceExecutor:
    def __init__(
            self,llm:Inference,query:str,
            prompt:str|None = None,
            context:str|None = None,
    ) -> None:
        self.llm = llm
        self.query = query
        self.context = context if context else ""

        if prompt is None:
            self.prompt = """
            You are a content creator. Write what the user is asking for while using the provided context as the primary source of information for the content.
            User Query: {query}
            Context: {context}
            """
        else:
            self.prompt = prompt
        
    def execute(self) -> str:
        self.llm.set_payload(
            inputs = self.prompt.format(query=self.query, context=self.context),
            parameters = {
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "repetition_penalty": 1.1,
                "temperature": settings.TEMPERATURE_INFERENCE
            }
        )
        answer = self.llm.inference()[0]["generated_text"]

        return answer
    
