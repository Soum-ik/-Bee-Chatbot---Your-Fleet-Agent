import os
from typing import List, Dict, Optional, Any, Union
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import requests
import json
from pydantic import Field
import tiktoken


class DeepSeekLLM(LLM):
    """Custom LangChain LLM wrapper for DeepSeek API with improved message handling"""
    
    api_key: str = Field(..., description="DeepSeek API key")
    model: str = Field(default="deepseek-chat", description="Model name")
    base_url: str = Field(default="https://api.deepseek.com/v1/chat/completions", description="API base URL")
    temperature: float = Field(default=0.7, description="Temperature for response generation")
    max_tokens: int = Field(default=1024, description="Maximum tokens in response")
    
    class Config:
        """Configuration for this pydantic object."""
        extra = "forbid"
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call DeepSeek API with the given prompt"""
        print(f"Prompt sent to LLM (length: {len(prompt)} chars):")
        print("-" * 40)
        print(prompt[:800] + ("..." if len(prompt) > 800 else ""))
        print("-" * 40)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add system message for better context handling in LangChain version too
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": """You are a helpful AI assistant with excellent contextual awareness. Always analyze the conversation history to understand connections between topics. 

Key guidelines:
- Look for logical connections between previous questions and current ones
- If someone asks about a product/vehicle first, then asks about routes/locations, assume they plan to use that product  
- Connect related topics intelligently (e.g., car specs → travel route = driving that car)
- Reference previous context when relevant to provide more helpful responses
- Be proactive in making useful connections between topics"""},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"Error calling DeepSeek API: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing API response: {str(e)}"


class ImprovedDeepSeekChatbot:
    """Enhanced chatbot with direct message management for better context control"""
    
    def __init__(self, api_key: str, max_history_messages: int = 20, max_tokens_per_request: int = 4000):
        self.api_key = api_key
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.temperature = 0.7
        self.max_tokens = 1024
        
        self.conversation_history = []
        self.max_history_messages = max_history_messages
        self.max_tokens_per_request = max_tokens_per_request
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            print("Warning: Could not load tiktoken. Token counting will be approximate.")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 4 characters per token
            return len(text) // 4
    
    def format_message_history(self, messages: List[Dict[str, str]]) -> str:
        """Format message history into a readable string"""
        if not messages:
            return "No previous conversation."
        
        formatted = []
        for msg in messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n\n".join(formatted)
    
    def trim_history_by_tokens(self, messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Trim conversation history to fit within token limit"""
        if not messages:
            return messages
        
        total_tokens = sum(self.count_tokens(msg["content"]) for msg in messages)
        
        while total_tokens > max_tokens and len(messages) > 2:
            # Remove oldest messages but keep the most recent exchange
            removed_msg = messages.pop(0)
            total_tokens -= self.count_tokens(removed_msg["content"])
        
        return messages
    
    def chat(self, message: str) -> str:
        """Send a message and get a response with proper context management"""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Manage history length and token count
        if len(self.conversation_history) > self.max_history_messages:
            self.conversation_history = self.conversation_history[-self.max_history_messages:]
        
        # Trim by tokens if necessary
        context_budget = self.max_tokens_per_request - 500  # Reserve tokens for response
        trimmed_history = self.trim_history_by_tokens(
            self.conversation_history[:-1],  # All except current message
            context_budget
        )
        
        # Add current message back
        current_context = trimmed_history + [self.conversation_history[-1]]
        
        try:
            response = self._call_api_with_messages(current_context)
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def _call_api_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """Call the DeepSeek API with a list of messages"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Enhanced system message for better contextual understanding
        system_message = """You are a helpful AI assistant with excellent contextual awareness. Follow these guidelines:

1. ANALYZE CONVERSATION FLOW: Always look for logical connections between previous questions and current ones
2. MAKE INTELLIGENT CONNECTIONS: If someone asks about a product/vehicle first, then asks about routes/locations, assume they plan to use that product
3. CONNECT RELATED TOPICS: Link related subjects intelligently (e.g., car specs → travel route = advice for driving that specific car)
4. REFERENCE CONTEXT: Always reference relevant previous context to provide more helpful, personalized responses
5. BE PROACTIVE: Anticipate user needs based on the conversation flow

Examples of good contextual responses:
- Car specs → Route planning = "For your Ford Mustang on this Toronto-Quebec route, here are fuel stops and performance considerations..."
- Restaurant question → Location question = "Based on the restaurant type you mentioned..."
- Product specs → Usage location = "For using this product in the location you mentioned..."

Always make these connections naturally and helpfully."""

        # Prepare messages for API
        api_messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history
        for msg in messages:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Debug: Print conversation context
        print(f"\nSending {len(api_messages)} messages to API:")
        for i, msg in enumerate(api_messages):
            role = msg["role"].title()
            content = msg["content"][:100] + ("..." if len(msg["content"]) > 100 else "")
            print(f"  {i+1}. {role}: {content}")
        print("-" * 50)
        
        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
    def save_conversation(self, filename: str):
        """Save conversation history to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
    
    def load_conversation(self, filename: str):
        """Load conversation history from a JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"Loaded {len(self.conversation_history)} messages from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except json.JSONDecodeError:
            print(f"Invalid JSON in file {filename}.")


class DeepSeekChatbot:
    """Enhanced chatbot class with improved context management using LangChain"""
    
    def __init__(self, api_key: str, max_token_limit: int = 2000):
        # Initialize the DeepSeek LLM
        self.llm = DeepSeekLLM(api_key=api_key)
        
        # Use ConversationSummaryBufferMemory for better token management
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=max_token_limit,
            memory_key="history",
            input_key="input",
            output_key="response",
            return_messages=True
        )
        
        # Improved prompt template with better contextual understanding
        self.prompt_template = PromptTemplate(
            input_variables=["history", "input"],
            template="""You are a helpful AI assistant with excellent contextual awareness. Always analyze the conversation history to understand connections between topics and provide relevant, contextual responses.

Key Instructions:
- Look for logical connections between previous questions and current ones
- If someone asks about a product/vehicle first, then asks about routes/locations, assume they plan to use that product
- Connect related topics intelligently (e.g., car specs → travel route = driving that car)
- Reference previous context when relevant to provide more helpful responses
- Be proactive in making useful connections between topics

Conversation History:
{history}

Current Question: {input}

Based on the conversation context above, provide a helpful and contextually aware response:"""
        )
        
        print("Prompt template initialized with improved formatting.")
        
        # Create the conversation chain with verbose mode for debugging
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt_template,
            verbose=True,  # Enable to see what's being sent
            output_key="response"
        )
    
    def chat(self, message: str) -> str:
        """Send a message to the chatbot and get a response"""
        try:
            response = self.conversation_chain.predict(input=message)
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history as a list of message dictionaries"""
        messages = self.memory.chat_memory.messages
        history = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        
        return history
    
    def clear_history(self):
        """Clear the conversation history"""
        self.memory.clear()
    
    def save_conversation(self, filename: str):
        """Save conversation history to a JSON file"""
        history = self.get_conversation_history()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def load_conversation(self, filename: str):
        """Load conversation history from a JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            self.clear_history()
            for entry in history:
                if entry["role"] == "human":
                    self.memory.chat_memory.add_user_message(entry["content"])
                elif entry["role"] == "assistant":
                    self.memory.chat_memory.add_ai_message(entry["content"])
                    
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except json.JSONDecodeError:
            print(f"Invalid JSON in file {filename}.")


def main():
    """Main function to run the chatbot with option to choose implementation"""
    
    # Get API key from environment variable or user input
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        api_key = input("Enter your DeepSeek API key: ").strip()
        if not api_key:
            print("API key is required to run the chatbot.")
            return
    
    # Choose implementation
    print("\nChoose chatbot implementation:")
    print("1. LangChain-based (with conversation summarization)")
    print("2. Direct API management (with token-aware trimming)")
    
    choice = input("Enter choice (1 or 2, default: 2): ").strip()
    
    if choice == "1":
        print("Initializing LangChain-based DeepSeek chatbot...")
        chatbot = DeepSeekChatbot(api_key=api_key, max_token_limit=2000)
    else:
        print("Initializing Direct API DeepSeek chatbot...")
        chatbot = ImprovedDeepSeekChatbot(
            api_key=api_key, 
            max_history_messages=20, 
            max_tokens_per_request=4000
        )
    
    print("\nChatbot is ready!")
    print("Commands:")
    print("- 'quit' or 'exit': Exit the chatbot")
    print("- 'clear': Clear conversation history")
    print("- 'save': Save conversation to file")
    print("- 'load': Load conversation from file")
    print("- 'history': Show conversation history")
    print("- 'stats': Show conversation statistics")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("Conversation history cleared.")
                continue
            elif user_input.lower() == 'save':
                filename = input("Enter filename to save (with .json extension): ").strip()
                if filename:
                    chatbot.save_conversation(filename)
                    print(f"Conversation saved to {filename}")
                continue
            elif user_input.lower() == 'load':
                filename = input("Enter filename to load (with .json extension): ").strip()
                if filename:
                    chatbot.load_conversation(filename)
                    print(f"Conversation loaded from {filename}")
                continue
            elif user_input.lower() == 'history':
                history = chatbot.get_conversation_history()
                if history:
                    print("\n" + "="*50)
                    print("CONVERSATION HISTORY")
                    print("="*50)
                    for i, entry in enumerate(history, 1):
                        role = "You" if entry["role"] in ["human", "user"] else "Assistant"
                        content = entry["content"]
                        print(f"\n{i}. {role}:")
                        print("-" * 20)
                        print(content)
                    print("="*50)
                else:
                    print("No conversation history.")
                continue
            elif user_input.lower() == 'stats':
                history = chatbot.get_conversation_history()
                if history:
                    total_messages = len(history)
                    user_messages = len([m for m in history if m["role"] in ["human", "user"]])
                    assistant_messages = len([m for m in history if m["role"] == "assistant"])
                    
                    print(f"\nConversation Statistics:")
                    print(f"- Total messages: {total_messages}")
                    print(f"- User messages: {user_messages}")
                    print(f"- Assistant messages: {assistant_messages}")
                    
                    if hasattr(chatbot, 'count_tokens'):
                        total_tokens = sum(chatbot.count_tokens(m["content"]) for m in history) # type: ignore
                        print(f"- Estimated total tokens: {total_tokens}")
                else:
                    print("No conversation history.")
                continue
            
            # Get response from chatbot
            print("Assistant: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()