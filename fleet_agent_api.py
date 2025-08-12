from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Sequence, Annotated
from functools import lru_cache
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque, defaultdict
from dotenv import load_dotenv

# LangChain imports for memory management
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# Import your existing prompts (assuming they exist)
try:
    from fleet_prompt import (
        FLEET_PROMPT, 
        FLEET_KEYWORDS, 
        NON_FLEET_INDICATORS, 
        FLEET_RESPONSE_INDICATORS,
        FLEET_CONTEXTS
    )
except ImportError:
    # Fallback definitions if module doesn't exist
    FLEET_PROMPT = "You are a fleet management expert."
    FLEET_KEYWORDS = ["fleet", "vehicle", "truck", "car", "maintenance", "fuel", "driver"]
    NON_FLEET_INDICATORS = ["weather", "cooking", "sports", "entertainment"]
    FLEET_RESPONSE_INDICATORS = ["fleet", "vehicle", "maintenance", "fuel", "driver"]
    FLEET_CONTEXTS = ["maintenance", "fuel", "route", "driver", "compliance"]

app = Flask(__name__)
CORS(app)

class UserIntent(Enum):
    """Enumeration of user intents for better categorization"""
    MAINTENANCE = "maintenance"
    OPTIMIZATION = "optimization" 
    ROUTE_PLANNING = "route_planning"
    FUEL_MANAGEMENT = "fuel_management"
    DRIVER_MANAGEMENT = "driver_management"
    COMPLIANCE = "compliance"
    DIAGNOSTICS = "diagnostics"
    FLEET_TRACKING = "fleet_tracking"
    COST_ANALYSIS = "cost_analysis"
    VEHICLE_SELECTION = "vehicle_selection"
    GENERAL_INQUIRY = "general_inquiry"

@dataclass
class ConversationTurn:
    """Data class for conversation turns with better structure"""
    role: str
    content: str
    timestamp: float
    topic: Optional[str] = None
    intent: Optional[UserIntent] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'topic': self.topic,
            'intent': self.intent.value if self.intent else None
        }

@dataclass
class SessionContext:
    """Optimized session context tracking"""
    primary_topic: Optional[str] = None
    last_intent: Optional[UserIntent] = None
    session_topics: set = None # type: ignore
    conversation_turns: int = 0
    
    def __post_init__(self):
        if self.session_topics is None:
            self.session_topics = set()

class ThreadedChatMessageHistory(BaseChatMessageHistory):
    """Thread-safe chat message history implementation with session support"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._messages: List[BaseMessage] = []
        self._lock = threading.Lock()
        self.max_messages = 40  # Limit to last 40 messages for efficiency
    
    @property
    def messages(self) -> List[BaseMessage]:
        with self._lock:
            return self._messages.copy()
    
    def add_message(self, message: BaseMessage) -> None:
        with self._lock:
            self._messages.append(message)
            # Maintain message limit (keep system message + recent messages)
            if len(self._messages) > self.max_messages:
                # Keep system message if it exists
                system_messages = [msg for msg in self._messages if isinstance(msg, SystemMessage)]
                other_messages = [msg for msg in self._messages if not isinstance(msg, SystemMessage)]
                
                # Keep most recent messages
                recent_messages = other_messages[-(self.max_messages - len(system_messages)):]
                self._messages = system_messages + recent_messages
    
    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
    
    def get_messages_by_type(self, message_type: type) -> List[BaseMessage]:
        """Get messages of specific type"""
        with self._lock:
            return [msg for msg in self._messages if isinstance(msg, message_type)]
    
    def get_last_n_messages(self, n: int) -> List[BaseMessage]:
        """Get last n messages"""
        with self._lock:
            return self._messages[-n:] if n > 0 else self._messages.copy()

class FleetChatMemory:
    """Lightweight memory manager storing only user questions for context"""
    
    def __init__(self):
        self.session_questions: Dict[str, List[str]] = {}
        self.session_contexts: Dict[str, SessionContext] = {}
        self._lock = threading.Lock()
        self.max_questions = 5  # Store last 5 questions for context
    
    def get_session_questions(self, session_id: str) -> List[str]:
        """Get or create session question history"""
        with self._lock:
            if session_id not in self.session_questions:
                self.session_questions[session_id] = []
                self.session_contexts[session_id] = SessionContext()
            return self.session_questions[session_id].copy()
    
    def get_session_context(self, session_id: str) -> SessionContext:
        """Get session context"""
        with self._lock:
            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = SessionContext()
            return self.session_contexts[session_id]
    
    def add_user_question(self, session_id: str, question: str, topic: Optional[str] = None, 
                         intent: Optional[UserIntent] = None) -> None:
        """Add user question to context history"""
        with self._lock:
            if session_id not in self.session_questions:
                self.session_questions[session_id] = []
                self.session_contexts[session_id] = SessionContext()
            
            questions = self.session_questions[session_id]
            context = self.session_contexts[session_id]
            
            # Add question and maintain limit
            questions.append(question)
            if len(questions) > self.max_questions:
                questions.pop(0)  # Remove oldest question
            
            # Update session context
            context.conversation_turns += 1
            if topic and topic != "fleet management":
                context.primary_topic = topic
                context.session_topics.add(topic)
            if intent:
                context.last_intent = intent
    
    def get_questions_context(self, session_id: str) -> str:
        """Get formatted previous questions as context"""
        questions = self.get_session_questions(session_id)
        if not questions:
            return ""
        
        context_parts = ["Previous questions in this conversation:"]
        for i, question in enumerate(questions, 1):
            context_parts.append(f"{i}. {question}")
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: str) -> None:
        """Clear specific session"""
        with self._lock:
            if session_id in self.session_questions:
                self.session_questions[session_id].clear()
                self.session_contexts[session_id] = SessionContext()
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary for session"""
        questions = self.get_session_questions(session_id)
        context = self.get_session_context(session_id)
        
        return {
            'session_id': session_id,
            'total_questions': len(questions),
            'conversation_turns': context.conversation_turns,
            'primary_topic': context.primary_topic,
            'last_intent': context.last_intent.value if context.last_intent else None,
            'session_topics': list(context.session_topics),
            'recent_questions': questions[-3:] if questions else []  # Last 3 questions
        }

class OptimizedFleetAgent:
    """Highly optimized fleet agent with LangChain memory integration"""
    
    def __init__(self):
        # Core configuration
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.deepseek_url = "https://api.deepseek.com/chat/completions"
        
        # LangChain memory system
        self.memory = FleetChatMemory()
        self.default_session_id = "default_session"  # Default session for backwards compatibility
        
        # Caching and performance optimization
        self._cache_lock = threading.Lock()
        self._response_cache = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        
        # Pre-compiled regex patterns for better performance
        self._compile_regex_patterns()
        
        # Fleet-related keywords and contexts (loaded once)
        self.fleet_keywords = set(FLEET_KEYWORDS + [
            # Add vehicle brands and models to fleet keywords
            'ford', 'mustang', 'toyota', 'camry', 'honda', 'civic', 'chevrolet', 'silverado',
            'nissan', 'altima', 'hyundai', 'elantra', 'bmw', 'mercedes', 'tesla', 'dodge',
            'jeep', 'ram', 'gmc', 'lexus', 'audi', 'specification', 'specs', 'details',
            'features', 'capacity', 'mpg', 'horsepower', 'performance', 'fuel economy',
            'dimensions', 'payload', 'towing', 'model', 'year', 'engine size', 'transmission'
        ])  # Set lookup is O(1)
        self.non_fleet_indicators = set(NON_FLEET_INDICATORS)
        self.fleet_response_indicators = set(FLEET_RESPONSE_INDICATORS)
        self.fleet_contexts = set(FLEET_CONTEXTS)
        
        # Intent classification patterns (pre-compiled)
        self._intent_patterns = self._build_intent_patterns()
        
        # Typo correction dictionary (loaded once)
        self._typo_corrections = self._build_typo_corrections()
    
    def _compile_regex_patterns(self):
        """Pre-compile frequently used regex patterns"""
        self.patterns = {
            'ambiguous': re.compile(r'\b(it|this|that|they|them|same|similar|previous|last|earlier)\b', re.IGNORECASE),
            'followup': re.compile(r'^(and|but|also|additionally|furthermore|what about|how about)\b', re.IGNORECASE),
            'short_question': re.compile(r'^\s*\w{1,4}\s*\??\s*$', re.IGNORECASE),
            'greeting': re.compile(r'\b(hi|hello|hey|good morning|good afternoon|good evening|greetings|howdy)\b', re.IGNORECASE)
        }
    
    def _build_intent_patterns(self) -> Dict[UserIntent, re.Pattern]:
        """Build pre-compiled regex patterns for intent classification"""
        intent_keywords = {
            UserIntent.MAINTENANCE: r'\b(maintain|service|repair|fix|check|inspect|replace|maintenance|servicing)\b',
            UserIntent.OPTIMIZATION: r'\b(optimize|improve|efficiency|performance|better|reduce cost|optimization)\b',
            UserIntent.ROUTE_PLANNING: r'\b(route|path|direction|navigate|travel|journey|routing|navigation)\b',
            UserIntent.FUEL_MANAGEMENT: r'\b(fuel|gas|consumption|mpg|efficiency|cost|fuel management)\b',
            UserIntent.DRIVER_MANAGEMENT: r'\b(driver|operator|staff|schedule|assignment|drivers)\b',
            UserIntent.COMPLIANCE: r'\b(compliance|regulation|legal|dot|safety|inspection|regulations)\b',
            UserIntent.DIAGNOSTICS: r'\b(problem|issue|sound|noise|symptom|diagnostic|troubleshoot)\b',
            UserIntent.FLEET_TRACKING: r'\b(track|monitor|location|gps|telematics|tracking)\b',
            UserIntent.COST_ANALYSIS: r'\b(cost|budget|expense|tco|financial|price|analysis)\b',
            UserIntent.VEHICLE_SELECTION: r'\b(choose|select|recommend|which vehicle|best vehicle|selection)\b'
        }
        
        return {intent: re.compile(pattern, re.IGNORECASE) for intent, pattern in intent_keywords.items()}
    
    def _build_typo_corrections(self) -> Dict[str, str]:
        """Build comprehensive typo correction dictionary"""
        return {
            # Vehicle-related typos
            'vehical': 'vehicle', 'vehichle': 'vehicle', 'vechile': 'vehicle',
            'maintenence': 'maintenance', 'maintainence': 'maintenance', 'maintance': 'maintenance',
            'breks': 'brakes', 'braeks': 'brakes', 'breakes': 'brakes',
            'engin': 'engine', 'engien': 'engine',
            'transmision': 'transmission', 'transmition': 'transmission',
            'stearing': 'steering', 'sterring': 'steering',
            'fllet': 'fleet', 'flet': 'fleet',
            'eficiency': 'efficiency', 'efficency': 'efficiency',
            'milage': 'mileage', 'mielage': 'mileage',
            'complience': 'compliance', 'compilance': 'compliance',
            # Add more as needed...
        }
    
    @lru_cache(maxsize=1000)
    def _classify_intent(self, question: str) -> UserIntent:
        """Efficiently classify user intent using cached regex matching"""
        question_lower = question.lower()
        
        best_intent = UserIntent.GENERAL_INQUIRY
        best_score = 0
        
        for intent, pattern in self._intent_patterns.items():
            matches = len(pattern.findall(question_lower))
            if matches > best_score:
                best_score = matches
                best_intent = intent
        
        return best_intent if best_score > 0 else UserIntent.GENERAL_INQUIRY
    
    @lru_cache(maxsize=500)
    def _extract_topic(self, question: str) -> str:
        """Efficiently extract topic using cached processing"""
        question_lower = question.lower()
        
        # Look for specific vehicle models/brands first
        vehicle_terms = []
        component_terms = []
        
        for keyword in self.fleet_keywords:
            if keyword in question_lower:
                if keyword in {'camry', 'corolla', 'prius', 'accord', 'civic', 'f-150', 
                              'silverado', 'mustang', 'tahoe', 'altima', 'elantra', 'optima', 
                              'toyota', 'honda', 'ford', 'chevrolet', 'chevy', 'nissan', 
                              'hyundai', 'bmw', 'mercedes', 'tesla', 'dodge', 'jeep'}:
                    vehicle_terms.append(keyword)
                elif keyword in {'engine', 'transmission', 'brake', 'steering', 'suspension', 
                               'battery', 'tire', 'oil', 'fuel', 'cooling', 'exhaust'}:
                    component_terms.append(keyword)
        
        if vehicle_terms:
            primary_vehicle = max(vehicle_terms, key=len)
            if component_terms:
                primary_component = max(component_terms, key=len)
                return f"{primary_vehicle} {primary_component}"
            return primary_vehicle
        
        if component_terms:
            return max(component_terms, key=len)
        
        return "fleet management"
    
    def _normalize_text_fast(self, text: str) -> str:
        """Fast text normalization using pre-built corrections"""
        if not text:
            return text
        
        # Apply typo corrections efficiently
        words = text.split()
        corrected_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            corrected_word = self._typo_corrections.get(word_lower, word)
            corrected_words.append(corrected_word)
        
        return ' '.join(corrected_words)
    
    @lru_cache(maxsize=200)
    def _is_greeting(self, question: str) -> bool:
        """Cached greeting detection"""
        return bool(self.patterns['greeting'].search(question))
    
    @lru_cache(maxsize=500)
    def _is_fleet_related(self, question: str) -> bool:
        """Enhanced fleet relation detection with better vehicle recognition"""
        question_lower = question.lower()
        
        # Quick rejection check for clearly non-fleet topics
        non_fleet_strict = {'weather', 'cooking', 'sports', 'music', 'movies', 'politics', 'health', 'dating'}
        if any(topic in question_lower for topic in non_fleet_strict):
            # Double check - if it mentions vehicles too, it might still be fleet related
            vehicle_terms = {'car', 'truck', 'vehicle', 'fleet', 'van', 'bus', 'suv', 'motorcycle'}
            if not any(term in question_lower for term in vehicle_terms):
                return False
        
        # ENHANCED: Vehicle specifications are fleet-related (for procurement decisions)
        specification_terms = {'specification', 'specs', 'details', 'features', 'capacity', 'mpg', 
                             'horsepower', 'engine', 'transmission', 'fuel economy', 'dimensions', 
                             'payload', 'towing', 'performance', 'model', 'year'}
        
        # Vehicle brands and models are fleet-related when asking for specs/info
        vehicle_brands = {'ford', 'toyota', 'honda', 'chevrolet', 'chevy', 'nissan', 'hyundai', 
                         'bmw', 'mercedes', 'tesla', 'dodge', 'jeep', 'ram', 'gmc', 'lexus', 
                         'audi', 'volkswagen', 'mazda', 'subaru', 'kia', 'volvo', 'acura'}
        
        vehicle_models = {'mustang', 'camry', 'corolla', 'prius', 'accord', 'civic', 'f-150', 
                         'silverado', 'tahoe', 'altima', 'elantra', 'optima', 'model s', 'model 3',
                         'escape', 'explorer', 'fusion', 'focus', 'ranger', 'bronco'}
        
        # Check for vehicle + specification context (this makes it fleet-related)
        has_vehicle_brand = any(brand in question_lower for brand in vehicle_brands)
        has_vehicle_model = any(model in question_lower for model in vehicle_models)
        has_spec_term = any(spec in question_lower for spec in specification_terms)
        
        # If asking about vehicle specifications, it's fleet-related
        if (has_vehicle_brand or has_vehicle_model) and (has_spec_term or 
            any(word in question_lower for word in ['of', 'about', 'for', 'specification', 'specs'])):
            return True
        
        # Check for direct fleet keywords
        if any(keyword in question_lower for keyword in self.fleet_keywords):
            return True
        
        # Check if it's a follow-up question with fleet context
        if (self.patterns['ambiguous'].search(question) or 
            self.patterns['followup'].search(question)) and self.session_context.primary_topic:
            return True
        
        # General vehicle terms with fleet contexts
        vehicle_terms = {'car', 'truck', 'vehicle', 'auto', 'fleet', 'van', 'bus', 'suv', 'motorcycle'}
        has_vehicle_term = any(term in question_lower for term in vehicle_terms)
        context_match = any(context in question_lower for context in self.fleet_contexts)
        
        # Simple vehicle questions (like "ford mustang") should be fleet-related
        if has_vehicle_brand or has_vehicle_model:
            return True
        
        return context_match and has_vehicle_term
    
    def _get_cache_key(self, question: str, context_summary: str) -> str:
        """Generate cache key for response caching"""
        return f"{hash(question)}_{hash(context_summary)}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        with self._cache_lock:
            if cache_key in self._response_cache:
                response, timestamp = self._response_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return response
                else:
                    del self._response_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache response with timestamp"""
        with self._cache_lock:
            self._response_cache[cache_key] = (response, time.time())
            
            # Clean old cache entries if cache gets too large
            if len(self._response_cache) > 1000:
                current_time = time.time()
                expired_keys = [
                    key for key, (_, timestamp) in self._response_cache.items()
                    if current_time - timestamp > self._cache_ttl
                ]
                for key in expired_keys:
                    del self._response_cache[key]
    
    def add_user_question(self, question: str, topic: Optional[str] = None, 
                         intent: Optional[UserIntent] = None, session_id: str = None):
        """Add user question to context memory"""
        if session_id is None:
            session_id = self.default_session_id
        
        self.memory.add_user_question(session_id, question, topic, intent)
    
    def get_session_context(self, session_id: str = None) -> SessionContext:
        """Get session context for backwards compatibility"""
        if session_id is None:
            session_id = self.default_session_id
        return self.memory.get_session_context(session_id)
    
    def _build_context_summary(self, session_id: str = None) -> str:
        """Build efficient context summary using question history"""
        if session_id is None:
            session_id = self.default_session_id
        
        context = self.memory.get_session_context(session_id)
        questions_context = self.memory.get_questions_context(session_id)
        
        context_parts = []
        
        # Add session information
        if context.primary_topic:
            context_parts.append(f"Primary Topic: {context.primary_topic}")
        
        if context.last_intent:
            context_parts.append(f"Last Intent: {context.last_intent.value}")
        
        # Add previous questions context
        if questions_context:
            context_parts.append(questions_context)
        
        return "\n".join(context_parts)
    
    def _enhance_question_with_context(self, question: str, session_id: str = None) -> str:
        """Enhance question with relevant context"""
        # Check if question needs context enhancement
        if (self.patterns['ambiguous'].search(question) or 
            self.patterns['short_question'].search(question) or
            len(question.split()) <= 4):
            
            context_summary = self._build_context_summary(session_id)
            if context_summary:
                return f"{question}\n\n[Context: {context_summary}]"
        
        return question
    
    def _get_deepseek_response(self, question: str, session_id: str = None) -> str: # pyright: ignore[reportArgumentType]
        """Optimized API call with LangChain memory integration"""
        if not self.deepseek_api_key:
            return "API key not configured"
        
        if session_id is None:
            session_id = self.default_session_id
        
        # Check cache first
        context_summary = self._build_context_summary(session_id)
        cache_key = self._get_cache_key(question, context_summary)
        cached_response = self._get_cached_response(cache_key)
        
        if cached_response:
            return cached_response
        
        # Build messages array with context-aware system prompt
        system_prompt = FLEET_PROMPT
        
        # Add previous questions context to system prompt
        questions_context = self.memory.get_questions_context(session_id)
        if questions_context:
            system_prompt += f"\n\n{questions_context}\n\nWhen answering the current question, consider the context from previous questions to provide relevant follow-up responses."
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add current enhanced question with fleet context instruction
        fleet_context_instruction = """
IMPORTANT: When users ask about vehicle specifications (like "Ford Mustang specifications"), provide comprehensive fleet-relevant details including:
- Engine specifications and fuel economy for fuel cost analysis
- Cargo/passenger capacity for operational planning  
- Maintenance intervals and costs for fleet budgeting
- Safety ratings for fleet risk management
- Total Cost of Ownership (TCO) considerations
- Fleet suitability assessment

Frame your response from a fleet management perspective, explaining how these specifications impact fleet operations."""
        
        enhanced_question = self._enhance_question_with_context(question, session_id)
        enhanced_question += f"\n\n{fleet_context_instruction}"
        
        messages.append({"role": "user", "content": enhanced_question})
        
        # API request with optimized payload
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "max_tokens": 10000,  # Increased for efficiency
            "temperature": 0.1,  # Slightly more deterministic
        }
        
        try:
            # Set timeout for better performance
            response = requests.post(
                self.deepseek_url, 
                headers=headers, 
                json=data, 
            )
            response.raise_for_status()
            
            result = response.json()
            ai_response = result['choices'][0]['message']['content']
            
            # Cache the response
            self._cache_response(cache_key, ai_response)
            
            return ai_response
            
        except requests.exceptions.Timeout:
            return "Request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"API communication error: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Unexpected API response format: {str(e)}"
    
    def process_question(self, question: str, session_id: str = None) -> Dict[str, Any]:
        """Main optimized question processing pipeline with LangChain memory"""
        start_time = time.time()
        
        if session_id is None:
            session_id = self.default_session_id
        
        # Step 1: Normalize text
        normalized_question = self._normalize_text_fast(question)
        
        # Step 2: Check for greeting
        if self._is_greeting(normalized_question):
            greeting_response = """Hey there! 👋 I'm Bee chatbot, your fleet agent with extensive experience in fleet operations, vehicle management, and logistics optimization.

What fleet management challenge can I help you solve today?"""
            
            self.add_user_question(question, session_id=session_id)
            
            return {
                'response': greeting_response,
                'agent_type': 'fleet_specialist',
                'validated': True,
                'processing_time': time.time() - start_time,
                'cached': False,
                'session_id': session_id
            }
        
        # Step 3: Check if fleet-related
        if not self._is_fleet_related(normalized_question):
            rejection_response = """Hey, I'm Bee chatbot, your fleet agent. That question doesn't seem related to fleet management. I specialize in:

🚛 Vehicle specifications & procurement decisions
👨‍💼 Driver management & scheduling  
🛣️ Route planning & logistics
🔧 Fleet maintenance & repairs
⛽ Fuel management & cost optimization
📱 Fleet tracking & telematics
📋 DOT compliance & safety
💰 Fleet budgeting & TCO analysis

If you're asking about vehicle specs for fleet purposes, please clarify! Otherwise, ask me about fleet operations."""
            
            return {
                'response': rejection_response,
                'rejected_topic': True,
                'agent_type': 'fleet_specialist',
                'processing_time': time.time() - start_time,
                'session_id': session_id
            }
        
        # Step 4: Extract topic and intent
        topic = self._extract_topic(normalized_question)
        intent = self._classify_intent(normalized_question)
        
        # Step 5: Add question to context
        self.add_user_question(question, topic, intent, session_id)
        
        # Step 6: Get AI response
        ai_response = self._get_deepseek_response(normalized_question, session_id)
        
        # Step 7: Response is not stored, only questions are kept for context
        
        # Step 8: Get session context for response metadata
        session_context = self.get_session_context(session_id)
        
        # Step 9: Return response with metadata
        return {
            'response': ai_response,
            'agent_type': 'fleet_specialist',
            'validated': True,
            'conversation_turn': session_context.conversation_turns,
            'primary_topic': session_context.primary_topic,
            'user_intent': intent.value,
            'session_topics': list(session_context.session_topics),
            'processing_time': time.time() - start_time,
            'cached': False,  # Could be enhanced to detect cache hits
            'session_id': session_id
        }
    
    def get_conversation_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Get conversation summary using LangChain memory"""
        if session_id is None:
            session_id = self.default_session_id
        
        return self.memory.get_conversation_summary(session_id)
    
    def clear_conversation(self, session_id: str = None):
        """Clear conversation using LangChain memory"""
        if session_id is None:
            session_id = self.default_session_id
        
        self.memory.clear_session(session_id)
        
        # Clear cache for this session
        with self._cache_lock:
            keys_to_remove = [key for key in self._response_cache.keys() if session_id in key]
            for key in keys_to_remove:
                del self._response_cache[key]
    
    def get_all_sessions(self) -> Dict[str, Any]:
        """Get summary of all active sessions"""
        sessions_info = {}
        for session_id in self.memory.session_questions.keys():
            sessions_info[session_id] = self.memory.get_conversation_summary(session_id)
        
        return {
            'total_sessions': len(self.memory.session_questions),
            'sessions': sessions_info,
            'cache_size': len(self._response_cache)
        }

# Initialize optimized agent
fleet_agent = OptimizedFleetAgent()

# Flask routes (simplified and optimized)
@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with session support"""
    try:
        data = request.get_json(force=True)  # force=True for better performance
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question in request body'}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Empty question provided'}), 400
        
        # Optional session ID for multi-user support
        session_id = data.get('session_id', None)
        
        # Process question through optimized pipeline
        result = fleet_agent.process_question(question, session_id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/conversation', methods=['GET'])
@app.route('/conversation/<session_id>', methods=['GET'])
def get_conversation(session_id=None):
    """Get conversation summary for specific session or default"""
    return jsonify(fleet_agent.get_conversation_summary(session_id))

@app.route('/sessions', methods=['GET'])
def get_all_sessions():
    """Get all active sessions summary"""
    return jsonify(fleet_agent.get_all_sessions())

@app.route('/clear', methods=['POST'])
@app.route('/clear/<session_id>', methods=['POST'])
def clear_conversation(session_id=None):
    """Clear conversation for specific session or default"""
    fleet_agent.clear_conversation(session_id)
    return jsonify({'message': f'Conversation cleared successfully for session: {session_id or "default"}'})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'agent': 'optimized_fleet_agent_with_langchain',
        'cache_size': len(fleet_agent._response_cache),
        'total_sessions': len(fleet_agent.memory.session_questions),
        'memory_system': 'Lightweight question-only context system'
    })

@app.route('/', methods=['GET'])
def home():
    """API documentation endpoint"""
    return jsonify({
        'message': 'Fleet Agent API with Question-Only Context',
        'version': '3.1',
        'features': [
            'Lightweight question-only context management',
            'Multi-session support with session IDs',
            'Thread-safe question history storage',
            'Response caching for improved performance',
            'Enhanced context tracking with metadata',
            'Efficient memory usage storing only questions'
        ],
        'endpoints': {
            'chat': 'POST /chat - Send question to fleet agent (supports session_id)',
            'conversation': 'GET /conversation[/<session_id>] - Get conversation summary',
            'sessions': 'GET /sessions - Get all active sessions',
            'clear': 'POST /clear[/<session_id>] - Clear conversation history',
            'health': 'GET /health - Check API health'
        },
        'usage_examples': {
            'basic_chat': {
                'method': 'POST',
                'url': '/chat',
                'body': {'question': 'How do I optimize fuel consumption?'}
            },
            'session_chat': {
                'method': 'POST', 
                'url': '/chat',
                'body': {'question': 'Tell me about maintenance', 'session_id': 'user_123'}
            }
        }
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)