import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-3B"

EVALUATION_TITLES = [
    "A Contrastive Objective for Cross-Lingual Dependency Parsing",
    "Retrieval-Augmented Morphological Tagging for Low-Resource Languages",
    "Efficient Prompting Strategies for Scientific Abstract Generation",
    "How much wood would a woodchuck if a woodchuck could chuck wood?",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    torch_dtype=torch.bfloat16,
)
model.eval()

for title in EVALUATION_TITLES:
    prompt = (
        "Write a computational linguistics research paper abstract for the following title.\n\n"
        f"Title:\n {title}\n\nAbstract:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"{text}\n\n")
