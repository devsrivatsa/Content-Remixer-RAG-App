import enum
from typing import Optional

from loguru import logger

try:
    import boto3
    from sagemaker.enums import EndpointType
    from sagemaker.huggingface import HuggingfaceModel
except ModuleNotFoundError:
    logger.warning("Couldn't load AWS Sagemaker imports. Run 'poetry install --with aws' to support AWS")

from ddd.domain.inference import DeploymentStrategy
from ddd.settings import settings

class SagemakerHuggingfaceStrategy(DeploymentStrategy):
    def __init__(self, deployment_service) -> None:
        """Initialized the deployment strategy with necessary services
        :param deployment_service: The service handling deployment details.
        :param logger: Logger for logging messages."""
        self.deployment_service = deployment_service
    
    def deploy(
            self, 
            role_arn:str, 
            llm_image:str,
            config:dict, 
            endpoint_name:str,
            endpoint_config_name:str,
            gpu_instance_type:str,
            resources:Optional[dict] = None,
            endpoint_type: enum.Enum = EndpointType.MODEL_BASED
    ) -> None:
        """Initiates the deployment process for a HuggingFace model on AWS SageMaker.

        :param role_arn: AWS role ARN with permissions for SageMaker deployment.
        :param llm_image: URI for the HuggingFace model Docker image.
        :param config: Configuration settings for the model environment.
        :param endpoint_name: Name of the SageMaker endpoint.
        :param endpoint_config_name: Name of the SageMaker endpoint configuration.
        :param resources: Optional resources for the model deployment (used for multi model endpoints)
        :param endpoint_type: can be EndpointType.MODEL_BASED (without inference component)
                or EndpointType.INFERENCE_COMPONENT (with inference component)

        """

        logger.info("Starting deployment using Sagemaker Huggingface strategy")
        logger.info("Deployment parameters: number of replicas: {settings.COPIES}, number of GPUs: {settings.GPUS}, instance type: {settings.GPU_INSTANCE_TYPE}")

        try:
            #deligate to deployment service to handle deployment details
            self.deployment_service.deploy(
                role_arn=role_arn,
                llm_image=llm_image,
                config=config,
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_config_name,
                gpu_instance_type=gpu_instance_type,
                resources=resources,
                endpoint_type=endpoint_type
            )
            logger.info("Deployment completed successfully")
        except Exception as e:
            logger.error(f"Error during deployment: {e}")
            raise e

class DeploymentService:
    def __init__(self, resource_manager) -> None:
        """Initialized the deployment service with necessary dependencies"""
        self.sagemaker_client = boto3.client(
            "sagemaker",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY
        )
        self.resource_manager = resource_manager


    def deploy(
        self,
        role_arn:str,
        llm_image:str,
        config:dict,
        endpoint_name:str,
        endpoint_config_name:str,
        gpu_instance_type:str,
        resources:Optional[dict] = None,
        endpoint_type: enum.Enum = EndpointType.MODEL_BASED
    ) -> None:
        try:
            if self.resource_manager.endpoint_config_exists(endpoint_config_name):
                logger.info(f"Endpoint configuration {endpoint_config_name} already exists")
            else:
                logger.info(f"Endpoint configuration {endpoint_config_name} does not exist.")
            
            self.prepare_and_deploy_model(
                role_arn=role_arn,
                llm_image=llm_image,
                config=config,
                endpoint_name=endpoint_name,
                gpu_instance_type=gpu_instance_type,
                resources=resources,
                update_endpoint=False,
                endpoint_type=endpoint_type
            )
            logger.info(f"Model deployed successfully to endpoint {endpoint_name}")
        except Exception as e:
            logger.error(f"Error during deployment: {e}")
            raise e
    
    @staticmethod
    def prepare_and_deploy_model(
        role_arn:str,
        llm_image:str,
        config:dict,
        endpoint_name:str,
        gpu_instance_type:str,
        resources:Optional[dict] = None,
        update_endpoint:bool = False,
        endpoint_type: enum.Enum = EndpointType.MODEL_BASED
    ) -> None:
        
        huggingface_model = HuggingfaceModel(
            role=role_arn,
            image_uri=llm_image,
            env=config,
        )
        huggingface_model.deploy(
            instance_type=gpu_instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            update_endpoint=update_endpoint,
            endpoint_type=endpoint_type,
            resources=resources,
            tags=[{"Key":"task", "Value":"model_task"}],
            container_startup_health_check_timeout=900
        )
