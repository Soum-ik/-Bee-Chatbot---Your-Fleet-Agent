
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepseek_chatbot import DeepSeekChatbot
import os

app = Flask(__name__)
CORS(app)

# Initialize chatbot with API key from environment variable
api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-e982d024e07740a3bbd074653523d60c"
chatbot = DeepSeekChatbot(api_key=api_key)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question in request body'}), 400
    user_message = data['question']
    response = chatbot.chat(user_message)
    return jsonify({'response': response})

@app.route('/history', methods=['GET'])
def history():
    history = chatbot.get_conversation_history()
    formatted = [
        {'role': 'User' if entry['role'] == 'human' else 'Bot', 'content': entry['content']}
        for entry in history
    ]
    return jsonify({'history': formatted})


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'DeepSeek Chatbot API',
        'endpoints': {
            'chat': 'POST /chat',
            'history': 'GET /history',
            'clear': 'POST /clear'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
