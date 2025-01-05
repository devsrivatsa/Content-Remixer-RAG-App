import os
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from zenml.client import Client
from zenml.exceptions import EntityExistsError
from dotenv import load_dotenv


class Settings(BaseSettings):
    def __init__(self) -> None:
        load_dotenv()
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # MongoDB
    DATABASE_HOST: str = "" #Field(default="mongodb://localhost:27017")
    DATABASE_NAME: str = "llm_twin"

    #AWS
    AWS_REGION: str = os.getenv("AWS_REGION")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SAGEMAKER_ROLE: str = os.getenv("AWS_SAGEMAKER_ROLE")

    #Huggingface api key
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")

    #AWS Sagemaker
    HF_MODEL_ID: str = "srivatsaHFHub/llama3.1_fineTomeAlpaca_modified_aligned"
    GPU_INSTANCE_TYPE: str = "ml.g5.2xlarge"
    SM_NUM_GPUS: int = 1
    MAX_INPUT_LENGTH: int = 2048
    MAX_TOTAL_TOKENS: int = 4096
    MAX_BATCH_TOTAL_TOKENS: int = 4096
    COPIES: int = 1
    GPUS: int = 1
    CPUS: int = 2
    MEMORY: int = 5 * 1024
    SAGEMAKER_ENDPOINT_CONFIG_INFERENCE: str = "llama3.1_fineTomeAlpaca_modified_aligned-inference"
    SAGEMAKER_ENDPOINT_INFERENCE: str = "llama3.1_fineTomeAlpaca_modified_aligned-inference"
    TEMPERATURE_INFERENCE: float = 0.01
    TOP_P_INFERENCE: float = 0.9
    MAX_NEW_TOKENS_INFERENCE: int = 150
    

settings = Settings()
