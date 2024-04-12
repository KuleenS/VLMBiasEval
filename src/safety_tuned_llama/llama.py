import os
import sys
from typing import List, Dict, Any

import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompter import Prompter

from config import SafetyTunedLLaMaConfig


class SafetyTunedLLaMa:

    def __init__(self, config: SafetyTunedLLaMaConfig) -> None:

        self.config = config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

        self.tokenizer.padding_side = "left"  # Allow batched inference

        self.prompter = Prompter(self.config.template_path)

        self.gradient_accumulation_steps = self.config.batch_size // self.config.micro_batch_size
        
    def tokenize(self, prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.config.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result 

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.config.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenize(
                user_prompt, add_eos_token=self.config.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if self.config.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def train(self):
        use_wandb = len(self.config.wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
        )
        # Only overwrite environ if wandb param passed
        if len(self.config.wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = self.config.wandb_project
        if len(self.config.wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = self.config.wandb_watch
        if len(self.config.wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = self.config.wandb_log_model

        model = prepare_model_for_int8_training(self.model)

        config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

        if self.config.data_path.endswith(".json") or self.config.data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=self.config.data_path)
        else:
            data = load_dataset(self.config.data_path)

        if resume_from_checkpoint:
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
                resume_from_checkpoint = (
                    False  # So the trainer won't try loading its state
                )
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")

        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

        if self.config.val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=self.config.val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle().map(self.generate_and_tokenize_prompt)
            val_data = None

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.config.micro_batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                warmup_steps=10,
                num_train_epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                fp16=True,
                logging_steps=10,
                optim="adamw_torch",
                evaluation_strategy="steps" if self.config.val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=25 if self.config.val_set_size > 0 else None,
                save_steps=25,
                output_dir=self.config.output_dir,
                save_total_limit=30,
                load_best_model_at_end=True if self.config.val_set_size > 0 else False,
                ddp_find_unused_parameters=False,
                group_by_length=self.config.group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name=self.config.wandb_run_name if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        model.save_pretrained(self.config.output_dir)





