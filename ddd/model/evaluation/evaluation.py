import os
import json
import gc
import concurrent.futures

from datasets import load_dataset, Dataset
from ..finetuning.finetune import inference, save_model, check_if_huggingface_model_exists
from openai import OpenAI
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DATASET_HUGGINGFACE_WORKSPACE = os.environ["DATASET_HUGGINGFACE_WORKSPACE"]
MODEL_HUGGINGFACE_WORKSPACE = os.environ["MODEL_HUGGINGFACE_WORKSPACE"]
IS_DUMMY = os.environ.get("IS_DUMMY", False)

def generate_answers(model_id:str, dataset_name:str):
    
    def apply_basic_chat_template(example):
        system_prompt = "You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."
        return {
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": example["instruction"],
                }
            ]
        }
    
    dataset = load_dataset(dataset_name, split="test")
    if IS_DUMMY:
        dataset = dataset.select(range(20))
    print(f"Dataset size: {len(dataset)}")
    dataset = dataset.map(apply_basic_chat_template)

    print(f"Generating answers for {model_id} on {dataset_name}...")
    llm = LLM(model=model_id, max_model_len=2048)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_p=0.05, max_tokens=2048)
    outputs = llm.generate(dataset["prompt"], sampling_params)
    
    answers = [output.outputs[0].text for output in outputs]
    dataset = dataset.add_column("answer", answers)

    print(f"Uploading results for {model_id} on {dataset_name}...")
    dataset.push_to_hub(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-results")
    
    gc.collect()

    return dataset

def evaluate_answer(instruction:str, answer:str, client:OpenAI) -> dict:
    prompt = f"""You are an expert judge. Please evaluate the quality of a given answer to an instruction based on two criteria:
    1. Accuracy: How factually correct is the information presented in the answer? You are a technical expert in this topic.
    2. Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.

    Accuracy Scale:
    1 (Poor): Contains factual errors or misleading information
    2 (Good): Mostly accurate with minor errors or omissions
    3 (Excellent): Highly accurate and comprehensive

    Style Scale:
    1 (Poor): Too formal, uses some overly complex words
    2 (Good): Good balance of technical content and accessibility, but still uses formal words and expressions
    3 (Excellent): Perfectly accessible language for blog/social media, uses simple but precise technical terms when necessary

    Example of bad style: The Llama2 7B model constitutes a noteworthy progression in the field of artificial intelligence, serving as the successor to its predecessor, the original Llama architecture.
    Example of excellent style: Llama2 7B outperforms the original Llama model across multiple benchmarks.

    Instruction: {instruction}

    Answer: {answer}

    Provide your evaluation in JSON format with the following structure:
    {{
        "accuracy": {{
            "analysis": "...",
            "score": 0
        }},
        "style": {{
            "analysis": "...",
            "score": 0
        }}
    }}
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role":"system",
                "content": "You are a helpful assistant who evaluates answers based on accuracy and style. Project your response in JSON format with a short analysis and a score for each criterion."
            },
            {"role": "user", "content": prompt}
        ],
        repponse_format = {"type":"json_object"},
        max_tokens=1000,
        temperature=0.9
    )

    return json.loads(completion.choices[0].message.content)

def evaluate_batch(batch, start_idx):
    client = OpenAI(api_key=OPENAI_API_KEY)
    return [(i, evaluate_answer(instruction, answer, client)) for i, (instruction, answer) in enumerate(batch, start=start_idx)]

def evaluate_answers(model_id:str, num_threads:int=10, batch_size:int=5) -> Dataset:
    #load the dataset
    dataset = load_dataset(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-results", split="test")

    #create batches of instruction answer paris
    batches = [
        (i, list(zip(dataset["instruction"][i:i+batch_size], dataset["answer"][i:i+batch_size], strict=False)))
        for i in range(0, len(dataset), batch_size)
    ]

    evaluations = [None] * len(dataset)

    #evaluate answers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(evaluate_batch, batch, start_idx) for start_idx, batch in batches]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            for idx, evaluation in future.result():
                evaluations[idx] = evaluation
    
    #replace evaluation column in dataset if present, else add it
    if "evaluation" in dataset.column_names:
        dataset = dataset.remove_columns(["evaluation"])
    dataset = dataset.add_column("evaluation", evaluations)

    #post process evaluations
    accuracy_scores = []
    style_scores = []

    for evaluation in dataset["evaluation"]:
        try:
            eval_dict = json.loads(evaluation) if isinstance(evaluation, str) else evaluation
            accuracy_scores.append(eval_dict["accuracy"]["score"])
            style_scores.append(eval_dict["style"]["score"])
        except (json.JSONDecodeError, KeyError, TypeError):
            accuracy_scores.append(None)
            style_scores.append(None)
    
    #add scores columns to dataset
    if "accuracy" in dataset.column_names:
        dataset = dataset.remove_columns(["accuracy"])
    dataset = dataset.add_column("accuracy", accuracy_scores)

    if "style" in dataset.column_names:
        dataset = dataset.remove_columns(["style"])
    dataset = dataset.add_column("style", style_scores)

    dataset.push_to_hub(f"{MODEL_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-evaluations")

    return dataset

def check_if_huggingface_dataset_exists(dataset_id:str, default_value:str) -> bool:
    api = HfApi()
    try:
        api.model_info(dataset_id)
        print(f"Found dataset {dataset_id} on HuggingFace")
    except RepositoryNotFoundError:
        print(f"Dataset {dataset_id} not found on HuggingFace.")
        dataset_id = default_value
        print(f"Using default dataset {dataset_id}")
    
    return dataset_id

model_ids = [
    check_if_huggingface_model_exists(f"{MODEL_HUGGINGFACE_WORKSPACE}/llama3.1_fineTomeAlpaca_modified", default_value="srivatsaHFHub/llama3.1_fineTomeAlpaca_modified"),
    check_if_huggingface_model_exists(f"{MODEL_HUGGINGFACE_WORKSPACE}/llama3.1_fineTomeAlpaca_modified_aligned", default_value="srivatsaHFHub/llama3.1_fineTomeAlpaca_modified_aligned"),
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
]

if __name__ == "__main__":
    for model_id in model_ids:
        dataset_name = check_if_huggingface_dataset_exists(f"{DATASET_HUGGINGFACE_WORKSPACE}/fineTomeAlpaca_modified", default_value="srivatsaHFHub/fineTomeAlpaca_modified")
        generate_answers(model_id, dataset_name)
    
    for model_id in model_ids:
        evaluate_answers(model_id)

    for model_id in model_ids:
        dataset = load_dataset(f"{DATASET_HUGGINGFACE_WORKSPACE}/{model_id.split('/')[-1]}-evaluations", split="all")
        accuracy_score = sum(dataset["accuracy"]) / len(dataset["accuracy"])
        style_score = sum(dataset["style"]) / len(dataset["style"])
        print(f"Model {model_id} accuracy score: {accuracy_score:.2f}")
        print(f"Model {model_id} style score: {style_score:.2f}")
