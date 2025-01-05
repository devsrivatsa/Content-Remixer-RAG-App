from loguru import logger
try:
    import boto3
    from botocore.exceptions import ClientError
except ModuleNotFoundError:
    logger.warning("Couldn't load AWS or sagemaker imports. Run 'poetry install --with aws' to support AWS.")

from ddd.settings import settings

class ResourceManager:
    def __init__(self) -> None:
        self.sagemaker_client = boto3.client(
            'sagemaker',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

    def endpoint_config_exists(self, endpoint_config_name:str) -> bool:
        """Check if an endpoint config exists"""
        try:
            self.sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            logger.info(f"Endpoint config {endpoint_config_name} exists")
            return True
        except ClientError:
            logger.info(f"Endpoint config {endpoint_config_name} does not exist")
            return False

    def endpoint_exists(self, endpoint_name:str) -> bool:
        """Check if an endpoint exists"""
        try:
            self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint {endpoint_name} exists")
            return True
        except ClientError:
            logger.info(f"Endpoint {endpoint_name} does not exist")
            return False
    