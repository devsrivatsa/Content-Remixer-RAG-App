import argparse
import os
from pathlib import Path
from unsloth import PatchDPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from typing import Any, List, Literal, Optional
import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import TextStreamer, TrainingArguments
from trl import DPOConfig, DPOTrainer, SFTTrainer


def load_model(
        model_name:str, 
        max_seq_len:int,
        load_in_4bit:bool,
        lora_r:int,
        lora_alpha:int,
        lora_dropout:float,
        target_modules:List[str],
        chat_template:str="llama-3"
) -> tuple:
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_len=max_seq_len,
        load_in_4bit=load_in_4bit
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template
    )

    return model, tokenizer

def load_and_prepare_dataset_for_sft(tokenizer):

        def apply_basic_chat_template(example):
            system_prompt = "You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": example["instruction"],
                    },
                    {
                        "role": "assistant",
                        "content": example["output"]
                    }
                ]
            }
        
        def formatting_prompts_func(examples):
            msgs = examples["messages"]
            texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in msgs]
            return {"text": texts}
        
        dataset1 = load_dataset("mlabonne/llmtwin")
        dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
        dataset = concatenate_datasets([dataset1["train"], dataset2]).remove_columns(["source", "score"])
        dataset = dataset.map(apply_basic_chat_template, remove_columns=["instruction", "output"])
        dataset = dataset.map(formatting_prompts_func, batched=True).remove_columns(["messages"])

        dataset = dataset.train_test_split(test_size=0.05)

        return dataset


def load_and_prepare_dataset_for_dpo(tokenizer):
    
    def apply_basic_preference_template(example):
        prompt = [
            {"role": "system", "content":"You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."},
            {"role": "user", "content": example["prompt"]},
        ]
        chosen = [{"role": "assistant", "content": example["chosen"]}]
        rejected = [{"role": "assistant", "content": example["rejected"]}]
        
        preference_example = {"prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected}
        
        return preference_example
    
    def formatting_prompts_func(examples):
        prompt = [tokenizer.apply_chat_template(txt, tokenize=False, add_generation_prompt=False) for txt in examples["prompt"]]
        chosen = [tokenizer.apply_chat_template(txt, tokenize=False, add_generation_prompt=False) for txt in examples["chosen"]]
        rejected = [tokenizer.apply_chat_template(txt, tokenize=False, add_generation_prompt=False) for txt in examples["rejected"]]
    
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    
    dpo_dataset = load_dataset("mlabonne/llmtwin-dpo", split="train")
    dpo_dataset = dpo_dataset.map(apply_basic_preference_template, remove_columns=["prompt", "rejected", "chosen"])
    dpo_dataset_formatted = dpo_dataset.map(formatting_prompts_func, batched=True)
    dpo_dataset = dpo_dataset_formatted.train_test_split(test_size=0.05)
    
    return dpo_dataset

def finetune(
        finetuning_type:Literal["sft", "dpo"],
        model_name:str,
        output_dir:str,
        dataset_huggingface_workspace:str,
        max_seq_len:int,
        load_in_4bit:bool=False,
        lora_r:int=32,
        lora_alpha:int=32,
        lora_dropout:float=0.0,
        target_modules:List[str] = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        chat_template:str = "llama3",
        learning_rate:float = 3e-4,
        num_train_epochs:int=3,
        per_device_train_batch_size:int = 2,
        gradient_accumulation_steps:int = 8,
        beta:float = 0.5,
        is_dummy:bool = True
) -> tuple:
    
    model, tokenizer = load_model(
        model_name,
        max_seq_len,
        load_in_4bit,
        lora_r,
        lora_alpha,
        lora_dropout,
        target_modules,
        chat_template
    )

    EOS_TOKEN = tokenizer.eos_token
    print(f"Setting EOS token to {EOS_TOKEN}")

    if is_dummy:
        num_train_epochs = 1
        print(f"Training in dummy mode. Setting num_train_epochs to '{num_train_epochs}'")
        print("Training in dummy mode. Reducing dataset size to 100 samples size to 400.")
    
    if finetuning_type == "sft":
        dataset = load_and_prepare_dataset_for_sft(tokenizer)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            dataset_text_field="text",
            max_seq_length=max_seq_len,
            dataset_num_proc=2,
            packing=True,
            args=TrainingArguments(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                per_device_eval_batch_size=per_device_train_batch_size,
                warmup_steps=10,
                output_dir=output_dir,
                report_to="comet_ml",
                seed=0,
            ),
        )
    elif finetuning_type == "dpo":
        dataset = load_and_prepare_dataset_for_dpo(tokenizer)
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            beta=beta,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            max_length=max_seq_len // 2,
            max_prompt_length=max_seq_len // 2,
            args=DPOConfig(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                per_device_eval_batch_size=per_device_train_batch_size,
                warmup_steps=10,
                output_dir=output_dir,
                eval_steps=0.2,
                logging_steps=1,
                report_to="comet_ml",
                seed=0,
            ),
        )
    else:
        raise ValueError(f"Invalid finetuning type: {finetuning_type}")

    trainer.train()

    return model, tokenizer


def inference(
    model:Any, 
    tokenizer:Any, 
    prompt:str = "Write a paragraph to introduce supervised fine tuning.", 
    max_new_tokens:int=256
) -> None:
    model = FastLanguageModel.for_inference(model)
    messages = [
        {"role": "system", "content":"You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request."},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
        torch_dtype = torch.bfloat16
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    model.eval()
    res = model.generate(
        input_ids = inputs,
        streamer = text_streamer,
        max_new_tokens = max_new_tokens,
        use_cache = True,
        temperature = 0.1,
        min_p = 0.1
    )
    print(res)

def save_model(model: Any, tokenizer: Any, output_dir: str, push_to_hub:bool=False, repo_id:Optional[str]=None):
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    if push_to_hub and repo_id:
        print(f"Saving model to '{repo_id}'")
        model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")

def check_if_huggingface_model_exists(model_id: str, default_value:str="srivatsaHFHub/llama3.1_fineTomeAlpaca_modified") -> str:
    api = HfApi()
    try:
        api.model_info(model_id)
    except RepositoryNotFoundError:
        print(f"Model '{model_id}' does not exist.")
        model_id = default_value
        print(f"Using default model '{model_id}'")

    return model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dataset_huggingface_workspace", type=str, default="mlabonne")
    parser.add_argument("--model_output_huggingface_workspace", type=str, default="mlabonne")
    parser.add_argument("--is_dummy", type=bool, default=False, help="Flag to reduce the dataset size for testing")
    parser.add_argument(
        "--finetuning_type",
        type=str,
        choices=["sft", "dpo"],
        default="sft",
        help="Parameter to choose the finetuning stage.",
    )

    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    print(f"Num training epochs: '{args.num_train_epochs}'")  # noqa
    print(f"Per device train batch size: '{args.per_device_train_batch_size}'")  # noqa
    print(f"Learning rate: {args.learning_rate}")  # noqa
    print(f"Datasets will be loaded from Hugging Face workspace: '{args.dataset_huggingface_workspace}'")  # noqa
    print(f"Models will be saved to Hugging Face workspace: '{args.model_output_huggingface_workspace}'")  # noqa
    print(f"Training in dummy mode? '{args.is_dummy}'")  # noqa
    print(f"Finetuning type: '{args.finetuning_type}'")  # noqa

    print(f"Output data dir: '{args.output_data_dir}'")  # noqa
    print(f"Model dir: '{args.model_dir}'")  # noqa
    print(f"Number of GPUs: '{args.n_gpus}'")  # noqa

    if args.finetuning_type == "sft":
        print("Starting SFT training...")  # noqa
        base_model_name = "meta-llama/Meta-Llama-3.1-8B"
        print(f"Training from base model '{base_model_name}'")  # noqa

        output_dir_sft = Path(args.model_dir) / "output_sft"
        model, tokenizer = finetune(
            finetuning_type="sft",
            model_name=base_model_name,
            output_dir=str(output_dir_sft),
            dataset_huggingface_workspace=args.dataset_huggingface_workspace,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=args.learning_rate,
        )
        inference(model, tokenizer)
        sft_output_model_repo_id = f"{args.model_output_huggingface_workspace}/llama3.1_fineTomeAlpaca_modified"
        save_model(model, tokenizer, "model_sft", push_to_hub=True, repo_id=sft_output_model_repo_id)
    
    elif args.finetuning_type == "dpo":
        print("Starting DPO training...")
        sft_base_model_repo_id = f"{args.model_output_huggingface_workspace}/llama3.1_fineTomeAlpaca_modified"
        sft_base_model_repo_id = check_if_huggingface_model_exists(sft_base_model_repo_id)
        print(f"Using SFT base model '{sft_base_model_repo_id}' for DPO training")

        output_dir_dpo = Path(args.model_dir)/"output_dpo"
        model, tokenizer = finetune(
            finetuning_type="dpo", 
            model_name=sft_base_model_repo_id,
            output_dir=str(output_dir_dpo),
            dataset_huggingface_workspace=args.dataset_huggingface_workspace,
            num_train_epochs=1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=2e-6,
            is_dummy=args.is_dummy
        )
        
        dpo_output_model_repo_id = f"{args.model_output_huggingface_workspace}/llama3.1_fineTomeAlpaca_modified_aligned"
        save_model(model, tokenizer, "model_dpo", push_to_hub=True, repo_id=dpo_output_model_repo_id)
