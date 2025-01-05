from loguru import logger
try:
    from sagemaker.enums import EndpointType
    from sagemaker.huggingface import get_huggingface_llm_image_uri
except ModuleNotFoundError:
    logger.warning("Couldn't load AWS Sagemaker imports. Run 'poetry install --with aws' to support AWS")

from ddd.model.utils import ResourceManager
from ddd.settings import settings
from .config import huggingface_deployment_config, model_resource_config
from .sagemaker_huggingface import DeploymentService, SagemakerHuggingfaceStrategy

def create_endpoint(endpoint_type:EndpointType.INFERENCE_COMPONENT_BASED) -> None:
    assert settings.AWS_ARN_ROLE is not None, "AWS_ARN_TOLE is not set in the .env file"
    logger.info(f"Creating endpoint with endpoint type {endpoint_type} and model id {settings.HF_MODEL_ID}")
    llm_image = get_huggingface_llm_image_uri("huggingface", version="2.2.0")
    resource_manager = ResourceManager()
    deployment_service = DeploymentService(resource_manager)

    SagemakerHuggingfaceStrategy(deployment_service).deploy(
        role_arn=settings.AWS_ARN_ROLE,
        llm_image=llm_image,
        config=huggingface_deployment_config,
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
        endpoint_config_name=settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE,
        gpu_instance_type=settings.GPU_INSTANCE_TYPE,
        resources=model_resource_config,
        endpoint_type=endpoint_type
    )


if __name__ == "__main__":
    create_endpoint(EndpointType.MODEL_BASED)