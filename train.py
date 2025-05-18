from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

# ✅ 模型和分词器
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# ✅ 加载聊天文本数据
with open("my_data_500_deduplicated.txt", "r", encoding="utf-8") as f:
    blocks = f.read().strip().split("\n\n")

# ✅ 构建数据集
data = [{"text": block.replace("\n", " ").strip()} for block in blocks if len(block.strip()) > 10]
dataset = Dataset.from_list(data)

def tokenize(example):
    result = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./chat-model",
    overwrite_output_dir=True,
    num_train_epochs=10,              # 多轮训练
    per_device_train_batch_size=4,    # 可以试试 4 或 8，训练更稳定
    save_steps=100,
    logging_steps=10,
    learning_rate=5e-5,               # 设置学习率，默认太低学习慢
    fp16=False                        # 如果你用的是 Mac Intel/ARM CPU
)



# ✅ 数据拼接方式（用于 Causal LM）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ✅ 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("🚀 Starting training...")
trainer.train()
print("✅ Training complete. Model saved to ./chat-model")
