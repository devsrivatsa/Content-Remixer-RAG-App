import json
from loguru import logger
try:
    from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
except ModuleNotFoundError:
    logger.warning("Couldn't load AWS Sagemaker imports. Run 'poetry install --with aws' to support AWS")

from ddd.settings import settings

huggingface_deployment_config = {
    "HF_MODEL_ID": settings.HF_MODEL_ID,
    "HUGGINGFACE_HUB_TOKEN": settings.HUGGINGFACE_HUB_TOKEN,
    "SM_NUM_GPUS": json.dumps(settings.SM_NUM_GPUS),
    "MAX_INPUT_LENGTH": json.dumps(settings.MAX_INPUT_LENGTH),
    "MAX_TOTAL_TOKENS": json.dumps(settings.MAX_TOTAL_TOKENS),
    "MAX_BATCH_TOTAL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "MAX_BATCH_PREFILL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "HF_MODEL_QUANTIZE": "bitsandbytes",
}

model_resource_config = ResourceRequirements(
    requests={
        "copies": settings.COPIES,
        "num_accelerators": settings.GPUS,
        "num_cpus": settings.NUM_CPUS,
        "memory": settings.MEMORY
    }
)