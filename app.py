from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})


# 重新加载你训练用的模型
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 保存到 ./chat-model（重新写入正确格式）
model.save_pretrained("./chat-model")
tokenizer.save_pretrained("./chat-model")

app = Flask(__name__)

# 聊天接口
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    output_ids = model.generate(
        input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"response": response})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
