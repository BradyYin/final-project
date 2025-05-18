from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ 使用全局 CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
CORS(app)  # ✅ 全局启用 CORS

# ✅ 加载模型
tokenizer = AutoTokenizer.from_pretrained("./chat-model")
model = AutoModelForCausalLM.from_pretrained("./chat-model")

@app.route("/chat", methods=["POST"])  # ❗只需要 POST，不要 OPTIONS
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    # ✅ 构造 prompt，与训练数据保持一致
    prompt = (
        "The following is a friendly conversation between a human and an AI assistant.\n"
        f"User: {message}\nAI:"
    )



    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # ✅ 控制输出长度，避免胡说或重复
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 60,  # 只生成回答部分
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = output_text.split("AI:", 1)[1].split("User:", 1)[0].strip()


    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
