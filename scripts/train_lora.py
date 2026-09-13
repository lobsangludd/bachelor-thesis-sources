import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

ROOT_PATH = Path(__file__).parent.parent
base, out, quant = sys.argv[1:4]

MAX_SEQ = 8192
TURN_END = {"gemma4_unified": "<turn|>\n", "gemma3": "<end_of_turn>\n", "gemma3_text": "<end_of_turn>\n"}


def load(path):
    rows = [json.loads(line)["messages"] for line in open(path, encoding="utf-8")]
    return [(messages[0]["content"], messages[1]["content"]) for messages in rows]


def encode(tokenizer, end, rows):
    examples = []
    for prompt, target in rows:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target + end, add_special_tokens=False)["input_ids"]

        if len(prompt_ids) >= MAX_SEQ:
            continue

        ids = (prompt_ids + target_ids)[:MAX_SEQ]
        labels = ([-100] * len(prompt_ids) + target_ids)[:MAX_SEQ]
        examples.append({"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)})

    return examples


def collate(batch):
    width = max(len(example["input_ids"]) for example in batch)
    pad = lambda key, value: torch.tensor([e[key] + [value] * (width - len(e[key])) for e in batch])

    return {"input_ids": pad("input_ids", tokenizer.pad_token_id), "labels": pad("labels", -100),"attention_mask": pad("attention_mask", 0)}


tokenizer = AutoTokenizer.from_pretrained(base)
end = TURN_END[AutoConfig.from_pretrained(base).model_type]
train, val = load(ROOT_PATH / "lora" / "lora_train_bare.jsonl"), load(ROOT_PATH / "lora" / "lora_val_bare.jsonl")

if quant == "nf4":
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
    model.enable_input_require_grads()

model.config.use_cache = False

train_enc, val_enc = encode(tokenizer, end, train), encode(tokenizer, end, val)

print(f"encoded train {len(train_enc)}, val {len(val_enc)}")

model = get_peft_model(model, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

trainer = Trainer(
    model=model, train_dataset=train_enc, eval_dataset=val_enc, data_collator=collate,
    args=TrainingArguments(
        output_dir=out + "-ckpt", per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, gradient_checkpointing=True, num_train_epochs=2,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=20, bf16=True,
        logging_steps=10, eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100,
        seed=0, report_to="none", remove_unused_columns=False))

trainer.train()

print("final val:", trainer.evaluate())

model.save_pretrained(out)
tokenizer.save_pretrained(out)
