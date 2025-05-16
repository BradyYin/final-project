from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

# 加载 tokenizer 和模型
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # ← 就是这一行！
model = AutoModelForCausalLM.from_pretrained(model_name)


# 加载你的聊天文本为单一字符串
with open("my_data.txt", "r", encoding="utf-8") as f:
    lines = f.read().split("\n\n")  # 每段对话分开

# 构建样本
data = [{"text": line.replace("\n", " ")} for line in lines if len(line.strip()) > 10]

# 使用 Hugging Face datasets 构建训练集
dataset = Dataset.from_list(data)

# 分词
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir="./chat-model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=100,
    logging_steps=10,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("🚀 Starting training...")
trainer.train()
print("✅ Training complete. Model saved to ./chat-model")
