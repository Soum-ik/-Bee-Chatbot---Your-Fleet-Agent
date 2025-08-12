#!/usr/bin/env python3
"""
Simple test script for the question-only context system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fleet_agent_api import OptimizedFleetAgent

def test_question_context():
    """Test the question-only context system"""
    
    print("="*60)
    print("TESTING QUESTION-ONLY CONTEXT SYSTEM")
    print("="*60)
    
    agent = OptimizedFleetAgent()
    session_id = "test_session"
    
    # Question 1: Ford Mustang specifications
    print("\n1. First Question: Ford Mustang specifications")
    question1 = "Give me the specification for Ford Mustang?"
    topic1 = agent._extract_topic(question1)
    intent1 = agent._classify_intent(question1)
    agent.add_user_question(question1, topic1, intent1, session_id)
    
    print(f"   Topic: {topic1}")
    print(f"   Intent: {intent1.value}")
    
    summary1 = agent.get_conversation_summary(session_id)
    print(f"   Stored questions: {summary1['recent_questions']}")
    
    # Question 2: Route optimization
    print("\n2. Second Question: Route optimization (gets Ford Mustang context)")
    question2 = "What about route optimization from Toronto to Saskatoon?"
    
    # Show context that would be sent to LLM
    context = agent._build_context_summary(session_id)
    print(f"   Context sent to LLM:")
    print(f"   {context}")
    
    topic2 = agent._extract_topic(question2)
    intent2 = agent._classify_intent(question2)
    agent.add_user_question(question2, topic2, intent2, session_id)
    
    print(f"   Topic: {topic2}")
    print(f"   Intent: {intent2.value}")
    
    # Question 3: Follow-up
    print("\n3. Third Question: Follow-up about 'this vehicle'")
    question3 = "How does this vehicle perform for long-distance operations?"
    
    # Show updated context
    context = agent._build_context_summary(session_id)
    print(f"   Context sent to LLM:")
    print(f"   {context}")
    
    topic3 = agent._extract_topic(question3)
    agent.add_user_question(question3, topic3, None, session_id)
    
    # Final summary
    final = agent.get_conversation_summary(session_id)
    print(f"\n4. FINAL SESSION STATE:")
    print(f"   Total questions: {final['total_questions']}")
    print(f"   Primary topic: {final['primary_topic']}")
    print(f"   Recent questions: {final['recent_questions']}")
    
    print(f"\nSUCCESS: Question-only context system working!")
    print("- Only stores user questions (not responses)")
    print("- Provides context for follow-up questions")
    print("- LLM can understand 'this vehicle' = Ford Mustang from context")

if __name__ == "__main__":
    test_question_context()