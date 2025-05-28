from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure the Gemini API
genai.configure(api_key='AIzaSyDPrdcBTqHr-6Excgf-on28M7a9_h_06Es')

# Initialize the model
model = genai.GenerativeModel('models/gemini-1.5-flash-8b')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Generate response from Gemini
        response = model.generate_content(user_message)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
