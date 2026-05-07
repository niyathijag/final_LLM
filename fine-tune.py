import math
from dataclasses import dataclass
import time
from typing import Any

import pandas as pd
import torch
from datasets import (
    Dataset,
    disable_progress_bar
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)
from transformers.utils import logging as hf_logging

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = "./qwen25-title-abstract-qlora"
MAX_LENGTH = 1024
SEED = 42
EVALUATION_TITLES = [
    "A Contrastive Objective for Cross-Lingual Dependency Parsing",
    "Retrieval-Augmented Morphological Tagging for Low-Resource Languages",
    "Efficient Prompting Strategies for Scientific Abstract Generation",
    "How much wood would a woodchuck if a woodchuck could chuck wood?",
]

# The following is a custom collator.  It is used to build batches during
# training.  In particular, it pads the various input sequences to have the
# same length so the sequences in a batch can be bundled into a single tensor.
@dataclass
class Causal_LM_Dynamic_Padding_Collator:
    tokenizer: Any
    label_pad_token_id: int = -100  # PyTorch uses -100 to indicate a token should 
                                    # be ignored during training.
    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]

            # Pad the text.  We also set values in the attention mask and label
            # that indicate the padding tokens are not to be used in computing
            # attention or loss.
            pad_len = max_length - len(input_ids)
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            padded_attention_mask = attention_mask + [0] * pad_len
            padded_labels = labels + [self.label_pad_token_id] * pad_len

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# We are viewing the data as prompt (the title) + response (the abstract).
# To make this structure clear to the LLM we structure the text so that
# title + abstract look like
#   <TITLE>
#   the title
#   </TITLE>
#   <ABSTRACT>
#   the abstract
#   </ABSTRACT>.

def build_prompt(title: str) -> str:
    return (
        "<TITLE>\n"
        f"{title.strip()}\n"
        "</TITLE>\n"
        "<ABSTRACT>\n"
    )


def build_target(abstract: str) -> str:
    return (
        f"{abstract.strip()}\n"
        "</ABSTRACT>"
    )


def preprocess_example(example, tokenizer, max_length):
    prompt = build_prompt(example["title"])
    target = build_target(example["abstract"])

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    # Attention is computed on the prompt + response, but only the
    # response figures in the loss calculation.  The model learns how,
    # given a title, it should generate an abstract.
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention_mask = attention_mask[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "length": len(input_ids),
    }


def main():
    print("Done with imports; beginning execution...", flush=True)

    # Set the seed of the pseudo-random number generators (random, numpy, torch, &c.)
    # for reproducible (?) results.
    set_seed(42)

    # Progress bars lead to weird output when redirecting output to a file.
    disable_progress_bar()
    hf_logging.disable_progress_bar()
    
    # Read the text data.
    print("Reading the data...", flush=True)
    with open("titles.txt", "r") as f:
        titles = f.read().split('\n')
#       titles = titles[:1000]  # For faster testing.
    with open("abstracts.txt", "r") as f:
        abstracts = f.read().split('\n')
#       abstracts = abstracts[:1000]

    df = pd.DataFrame({"title": titles, "abstract": abstracts})
    dataset = Dataset.from_pandas(df, preserve_index=False)

    # Load the tokenizer.
    print("Loading the tokenizer...", flush=True)    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # eos = end of sequence.

    special_tokens = {
        "additional_special_tokens": [
            "<TITLE>",
            "</TITLE>",
            "<ABSTRACT>",
            "</ABSTRACT>",
        ]
    }
    num_tokens_added = tokenizer.add_special_tokens(special_tokens)

    # We use a tripartite partition of the data into training, validation, and evaluation sets.
    # Validation during training can be surprisingly expensive compared to training, because
    # validation is done using the entire validation set, while training steps are done with
    # much smaller batches.  For this reason we create a relatively small validation set and
    # reserve most of the held-out data for the evaluation of the final model.
    
    print("Splitting the data in to training, monitoring, and evaluation sets...", flush=True)        
    split = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_ds = split["train"]
    eval_ds = split["test"]
    split = eval_ds.train_test_split(test_size=500, seed=SEED)
    valid_ds = split["test"]

    # Preprocess the data.
    print("Tokenizing...", flush=True)
    begin = time.time()
    train_tok = train_ds.map(
        lambda x: preprocess_example(x, tokenizer, MAX_LENGTH),
        remove_columns=train_ds.column_names,
    )
    eval_tok = eval_ds.map(
        lambda x: preprocess_example(x, tokenizer, MAX_LENGTH),
        remove_columns=eval_ds.column_names,
    )
    valid_tok = valid_ds.map(
        lambda x: preprocess_example(x, tokenizer, MAX_LENGTH),
        remove_columns=valid_ds.column_names,
    )
    end = time.time()
    print(f"Tokenization required {end-begin} seconds...")

    # Load the quantized base model.
    print("Loading the model...", flush=True)    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,  # Brain float has a wider range of magnitudes than IEEE 754 binary16.
    )
    print(model)

    if num_tokens_added > 0:  # Needed because we added special tokens to the vocabulary.
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Merge LoRA with the model.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training setup.
    print("Training setup...", flush=True)                
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        disable_tqdm=True,  # Disable the progress bar for background runs.
        # The following requires 25 GB of GPU memory.
        per_device_train_batch_size=8,  #  8 (25 GB) 
        gradient_accumulation_steps=16, # 16 (25 GB)
        # The following requires 18 GB of GPU memory.
#        per_device_train_batch_size=4,  #  4 (18 GB)
#        gradient_accumulation_steps=32, # 32 (18 GB)
        num_train_epochs= 3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=1000,
#        per_device_eval_batch_size=1
        save_steps=1000,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=500,
        weight_decay=0.01,
        report_to="none",
        load_best_model_at_end=True,
        length_column_name="length",
        train_sampling_strategy="group_by_length",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    data_collator = Causal_LM_Dynamic_Padding_Collator(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=data_collator,
#        callbacks=[sample_callback],
    )

    # Train.
    print("Training...", flush=True)
    trainer.train()

    # Evaluate on the full test set.
    print("Evaluating the final model...", flush=True)
    eval_results = trainer.evaluate(eval_dataset=eval_tok)
    print(eval_results)
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    print("eval_loss:", eval_loss)
    print("perplexity:", perplexity)

    # Save the model and its tokenizer.
    print("Saving model...", flush=True)
    model.save_pretrained(OUTPUT_DIR)  # This really only saves the LoRA information!
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Reload for inference.
    print("Reloading model...", flush=True)    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    if num_tokens_added > 0:
        base_model.resize_token_embeddings(len(tokenizer))

    # Combine the base model with the LoRA updates saved in OUTPUT_DIR.
    inference_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    inference_model.eval()  # Switch the model from training to inference mode.

    # Generate abstracts.
    end_abstract_id = tokenizer.convert_tokens_to_ids("</ABSTRACT>")
    eos_ids = [tokenizer.eos_token_id, end_abstract_id]
    for new_title in EVALUATION_TITLES:
        prompt = build_prompt(new_title)
        inputs = tokenizer(prompt, return_tensors="pt").to(inference_model.device)
        with torch.no_grad():
            outputs = inference_model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )
            
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated_part = full_text[len(prompt):]
        abstract = generated_part.split("</ABSTRACT>")[0].strip()

        print("\nTITLE:\n", new_title)
        print("\nGENERATED ABSTRACT:\n", abstract)


if __name__ == "__main__":
    main()
