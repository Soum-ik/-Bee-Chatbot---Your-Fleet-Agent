from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
from typing import Dict, Any
from fleet_prompt import (
    FLEET_PROMPT, 
    FLEET_KEYWORDS, 
    NON_FLEET_INDICATORS, 
    FLEET_RESPONSE_INDICATORS,
    FLEET_CONTEXTS
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class FleetAgent:
    def __init__(self):
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.deepseek_url = "https://api.deepseek.com/chat/completions"
        
        # Import keywords and prompts from separate file
        self.fleet_keywords = FLEET_KEYWORDS
        self.non_fleet_indicators = NON_FLEET_INDICATORS
        self.fleet_prompt = FLEET_PROMPT
        self.fleet_response_indicators = FLEET_RESPONSE_INDICATORS
        self.fleet_contexts = FLEET_CONTEXTS
        
        # Conversation history storage
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 messages for context
    
    def add_to_history(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({
            'role': role,
            'content': content
        })
        
        # Keep only the last N messages
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for the prompt"""
        if not self.conversation_history:
            return ""
        
        context = "\n\nCONVERSATION HISTORY (for context only):\n"
        for message in self.conversation_history:
            role = "User" if message['role'] == 'user' else "Assistant"
            context += f"{role}: {message['content']}\n"
        
        return context + "\n"
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
    def is_greeting(self, question: str) -> bool:
        """Check if the user's message is a greeting"""
        greeting_keywords = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'greetings', 'howdy', 'whats up', "what's up", 'sup', 'yo',
            'good day', 'morning', 'afternoon', 'evening'
        ]
        
        question_lower = question.lower().strip()
        return any(greeting in question_lower for greeting in greeting_keywords)
    
    def get_greeting_response(self) -> str:
        """Return a friendly greeting response"""
        return """Hey there! 👋 I'm Bee chatbot, your fleet agent with extensive experience in fleet operations, vehicle management, and logistics optimization.

What fleet management challenge can I help you solve today?"""

    def is_fleet_related(self, question: str) -> bool:
        question_lower = question.lower()
        
        # Check for non-fleet indicators first (more restrictive)
        if any(indicator in question_lower for indicator in self.non_fleet_indicators):
            return False
            
        # Check for fleet keywords
        fleet_match = any(keyword in question_lower for keyword in self.fleet_keywords)
        
        # Additional context analysis
        context_match = any(context in question_lower for context in self.fleet_contexts)
        
        # Broader vehicle-related terms for inclusive detection
        vehicle_terms = ['car', 'truck', 'vehicle', 'auto', 'fleet', 'van', 'bus', 'motorcycle', 'suv']
        has_vehicle_term = any(term in question_lower for term in vehicle_terms)
        
        # Question is fleet-related if:
        # 1. Direct fleet keyword match, OR
        # 2. Context match with any vehicle term, OR
        # 3. Context match with traditional fleet/vehicle terms
        return (fleet_match or 
                (context_match and has_vehicle_term) or 
                (context_match and ('fleet' in question_lower or 'vehicle' in question_lower)))
    
    def validate_response_is_fleet_related(self, response: str) -> bool:
        """Additional validation to ensure the AI response is fleet-related"""
        response_lower = response.lower()
        
        # Check if response contains rejection message
        if "fleet management agent" in response_lower and "can only assist" in response_lower:
            return False
            
        # Check if response contains fleet-related terms
        return any(indicator in response_lower for indicator in self.fleet_response_indicators)

    def get_deepseek_response(self, prompt: str) -> str:
        if not self.deepseek_api_key:
            return "API key not configured"
            
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        # Get conversation context
        conversation_context = self.get_conversation_context()
        
        # Enhanced prompt with double-check and conversation context
        enhanced_prompt = f"""FLEET QUESTION: {prompt}
        
{conversation_context}Remember: You are a Fleet Management Expert. Only answer if this is about fleet operations, vehicles, drivers, logistics, maintenance, fuel, tracking, compliance, or fleet costs others related to fleet management/ any other topic related to fleet management. If not fleet-related, respond with the rejection message."""
        
        # Prepare messages for the API
        messages = [{"role": "system", "content": self.fleet_prompt}]
        
        # Add conversation history to maintain context
        for message in self.conversation_history:
            messages.append(message)
        
        # Add the current user message
        messages.append({"role": "user", "content": enhanced_prompt})
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.deepseek_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            return f"Error communicating with DeepSeek API: {str(e)}"
        except KeyError as e:
            return f"Unexpected API response format: {str(e)}"

fleet_agent = FleetAgent()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing question in request body'
            }), 400
            
        question = data['question']
        
        if fleet_agent.is_greeting(question):
            greeting_response = fleet_agent.get_greeting_response()
            # Add greeting and response to history
            fleet_agent.add_to_history("user", question)
            fleet_agent.add_to_history("assistant", greeting_response)
            
            return jsonify({
                'response': greeting_response,
                'agent_type': 'fleet_specialist',
                'validated': True
            })

        if not fleet_agent.is_fleet_related(question):
            return jsonify({
                'response': "Hey, I'm Bee chatbot, your fleet agent, and I can't answer those questions. I'm specialized ONLY in fleet management topics like:\n\n🚛 Vehicle fleet operations & optimization\n👨‍💼 Driver management & scheduling\n🛣️ Route planning & logistics\n🔧 Fleet maintenance & repairs\n⛽ Fuel management & cost optimization\n📱 Fleet tracking & telematics\n📋 DOT compliance & safety\n💰 Fleet budgeting & TCO analysis\n\nPlease ask me something about fleet operations, and I'll provide expert guidance!",
                'rejected_topic': True,
                'agent_type': 'fleet_specialist'
            })
        
        # Add user message to history
        fleet_agent.add_to_history("user", question)
        
        response = fleet_agent.get_deepseek_response(question)
        
        # Double-check that the AI response is actually fleet-related
        if not fleet_agent.validate_response_is_fleet_related(response):
            # Add rejected response to history for context
            fleet_agent.add_to_history("assistant", response)
            
            return jsonify({
                'response': "I'm Bee chatbot, your specialized fleet agent, and I can only assist with fleet operations, vehicle management, driver coordination, and related logistics topics. Please ask me about fleet management instead!",
                'rejected_by_validation': True,
                'agent_type': 'fleet_specialist'
            })
        
        # Add valid response to history
        fleet_agent.add_to_history("assistant", response)
        
        return jsonify({
            'response': response,
            'agent_type': 'fleet_specialist',
            'validated': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'agent': 'fleet_agent'
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Fleet Agent API',
        'endpoints': {
            'chat': 'POST /chat - Send a question to the fleet agent',
            'health': 'GET /health - Check API health'
        },
        'usage': {
            'chat_example': {
                'method': 'POST',
                'url': '/chat',
                'body': {'question': 'How do I optimize fuel consumption for my fleet?'}
            }
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)