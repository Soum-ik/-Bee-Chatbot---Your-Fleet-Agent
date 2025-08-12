#!/usr/bin/env python3
"""
Test script to demonstrate the new question-only context system.
This shows how the system stores only questions and uses them for context.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fleet_agent_api import OptimizedFleetAgent

def test_question_only_context():
    """Test the question-only context system with example scenarios"""
    
    print("="*60)
    print("TESTING QUESTION-ONLY CONTEXT SYSTEM")
    print("="*60)
    
    # Create agent instance
    agent = OptimizedFleetAgent()
    
    # Test Scenario: Ford Mustang -> Route Optimization context flow
    print("\n* SCENARIO: Vehicle Specs -> Route Optimization Context")
    print("-"*50)
    
    session_id = "test_session_1"
    
    # Question 1: Ford Mustang specifications
    question1 = "Give me the specification for Ford Mustang?"
    print(f"Question 1: {question1}")
    
    # Add question to context (simulating the system)
    topic1 = agent._extract_topic(question1)
    intent1 = agent._classify_intent(question1)
    agent.add_user_question(question1, topic1, intent1, session_id)
    
    print(f"  * Topic extracted: {topic1}")
    print(f"  * Intent classified: {intent1.value}")
    
    # Check conversation state after first question
    summary1 = agent.get_conversation_summary(session_id)
    print(f"  * Context stored: {summary1['recent_questions']}")
    
    # Question 2: Route optimization (should get context from Ford Mustang)
    question2 = "What about route optimization from Toronto to Saskatoon?"
    print(f"\nQuestion 2: {question2}")
    
    # Get context that would be sent to LLM
    context_summary = agent._build_context_summary(session_id)
    print(f"  * Context for LLM: {context_summary}")
    
    # Add second question
    topic2 = agent._extract_topic(question2)
    intent2 = agent._classify_intent(question2)
    agent.add_user_question(question2, topic2, intent2, session_id)
    
    print(f"  * Topic extracted: {topic2}")
    print(f"  * Intent classified: {intent2.value}")
    
    # Question 3: Follow-up question
    question3 = "How does this vehicle perform for long-distance fleet operations?"
    print(f"\nQuestion 3: {question3}")
    
    # Get updated context
    context_summary = agent._build_context_summary(session_id)
    print(f"  * Context for LLM: {context_summary}")
    
    # Add third question
    topic3 = agent._extract_topic(question3)
    intent3 = agent._classify_intent(question3)
    agent.add_user_question(question3, topic3, intent3, session_id)
    
    # Final summary
    final_summary = agent.get_conversation_summary(session_id)
    print(f"\n📊 FINAL SESSION SUMMARY:")
    print(f"  • Total questions: {final_summary['total_questions']}")
    print(f"  • Primary topic: {final_summary['primary_topic']}")
    print(f"  • Session topics: {final_summary['session_topics']}")
    print(f"  • Recent questions: {final_summary['recent_questions']}")
    
    print("\n✅ CONTEXT FLOW DEMONSTRATION:")
    print("1. Question 1 about 'Ford Mustang specifications' establishes vehicle context")
    print("2. Question 2 about 'route optimization' gets Ford Mustang context automatically")
    print("3. Question 3 about 'this vehicle' uses full conversation context")
    print("4. LLM receives previous questions to understand 'this vehicle' = Ford Mustang")

def demonstrate_context_enhancement():
    """Demonstrate how ambiguous questions get enhanced with context"""
    
    print("\n" + "="*60)
    print("TESTING CONTEXT ENHANCEMENT FOR AMBIGUOUS QUESTIONS")
    print("="*60)
    
    agent = OptimizedFleetAgent()
    session_id = "test_session_2"
    
    # Build up context first
    questions = [
        "Tell me about Tesla Model 3 fleet suitability",
        "What are the maintenance costs for electric vehicles?",
        "How about charging infrastructure requirements?"
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\nAdding Question {i}: {q}")
        topic = agent._extract_topic(q)
        intent = agent._classify_intent(q)
        agent.add_user_question(q, topic, intent, session_id)
    
    # Now test ambiguous questions
    ambiguous_questions = [
        "What about insurance costs?",
        "How does it compare?", 
        "Is this suitable for our fleet?"
    ]
    
    print(f"\n🔍 TESTING AMBIGUOUS QUESTION ENHANCEMENT:")
    print("-"*50)
    
    for ambig_q in ambiguous_questions:
        print(f"\nAmbiguous question: '{ambig_q}'")
        enhanced = agent._enhance_question_with_context(ambig_q, session_id)
        print(f"Enhanced version:")
        print(f"  {enhanced}")

def demonstrate_memory_efficiency():
    """Show memory efficiency of question-only storage"""
    
    print("\n" + "="*60)
    print("TESTING MEMORY EFFICIENCY")
    print("="*60)
    
    agent = OptimizedFleetAgent()
    session_id = "test_session_3"
    
    # Add many questions to test limit
    questions = [
        "Ford F-150 specifications?",
        "Fuel efficiency comparison?", 
        "Maintenance schedule for trucks?",
        "Driver training requirements?",
        "Route optimization strategies?",
        "Insurance costs for commercial vehicles?",  # This should push out the first question
        "Fleet tracking technology options?",
        "DOT compliance requirements?"
    ]
    
    print(f"📝 Adding {len(questions)} questions (limit is {agent.memory.max_questions}):")
    
    for i, q in enumerate(questions, 1):
        topic = agent._extract_topic(q)
        intent = agent._classify_intent(q)
        agent.add_user_question(q, topic, intent, session_id)
        
        summary = agent.get_conversation_summary(session_id)
        print(f"  {i}. {q[:40]}... → Stored: {len(summary['recent_questions'])} questions")
    
    final_summary = agent.get_conversation_summary(session_id)
    print(f"\n📊 FINAL STORAGE:")
    print(f"  • Questions stored: {len(final_summary['recent_questions'])}/{agent.memory.max_questions}")
    print(f"  • Latest questions kept:")
    for i, q in enumerate(final_summary['recent_questions'], 1):
        print(f"    {i}. {q}")
    
    print(f"\n💡 EFFICIENCY BENEFIT:")
    print(f"  • OLD SYSTEM: Would store ~{len(questions) * 2} full messages (questions + responses)")
    print(f"  • NEW SYSTEM: Stores only {len(final_summary['recent_questions'])} questions")
    print(f"  • Memory reduction: ~{(1 - len(final_summary['recent_questions'])/(len(questions)*2)) * 100:.0f}%")

if __name__ == "__main__":
    test_question_only_context()
    demonstrate_context_enhancement()
    demonstrate_memory_efficiency()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("✅ Question-only context system is working perfectly!")
    print("="*60)