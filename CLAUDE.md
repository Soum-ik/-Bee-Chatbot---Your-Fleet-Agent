# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a specialized Fleet Management API chatbot called "Bee chatbot" built with Flask and powered by DeepSeek AI. The application provides expert guidance on fleet operations, vehicle diagnostics, maintenance, and logistics optimization. It's designed to only respond to fleet-related queries and includes topic filtering to ensure responses stay within the fleet management domain.

## Architecture

The project consists of three main components:

1. **Flask API Server** (`fleet_agent_api.py`): Main application with conversation management and DeepSeek integration
2. **Prompt System** (`fleet_prompt.py`): Contains specialized fleet prompts, keywords, and topic filtering logic
3. **Web Interface** (`index.html`): Frontend chat interface for interacting with the fleet agent

### Key Components

- **FleetAgent Class** (`fleet_agent_api.py:17-160`): Core chatbot logic with conversation history, topic validation, and AI integration
- **Topic Filtering**: Multi-layer validation using fleet keywords, non-fleet indicators, and response validation
- **Conversation Management**: Maintains chat history with configurable limits for context preservation
- **DeepSeek Integration**: Handles API communication with proper error handling and response validation

## Common Development Commands

### Running the Application

**Local Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export DEEPSEEK_API_KEY=your_api_key_here  # Windows: set DEEPSEEK_API_KEY=your_api_key_here

# Run the application
python fleet_agent_api.py
```

**Docker Development:**
```bash
# Build and run with Docker Compose (recommended)
docker-compose up --build

# Manual Docker build
docker build -t fleet-agent-api .
docker run -p 5000:5000 -e DEEPSEEK_API_KEY=your_api_key_here fleet-agent-api

# View Docker logs
docker-compose logs
```

### Testing the API

```bash
# Health check
curl http://localhost:5000/health

# Test chat endpoint
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I optimize fuel consumption for my fleet?"}'

# View conversation history
curl http://localhost:5000/conversation-history

# Clear conversation history
curl -X POST http://localhost:5000/clear-history
```

## API Endpoints

- `POST /chat` - Send questions to the fleet agent
- `GET /health` - API health status
- `GET /conversation-history` - View chat history
- `POST /clear-history` - Clear chat history
- `GET /` - API information and usage examples

## Environment Variables

Required:
- `DEEPSEEK_API_KEY` - Your DeepSeek API key for AI responses

Optional:
- `FLASK_ENV` - Flask environment (production/development)
- `FLASK_DEBUG` - Enable Flask debug mode

## Fleet Topic System

The application uses a sophisticated topic filtering system:

1. **Fleet Keywords** (`fleet_prompt.py:50-177`): Extensive list of fleet, vehicle, and automotive terms
2. **Non-Fleet Indicators** (`fleet_prompt.py:180-187`): Terms that indicate non-fleet topics
3. **Response Validation** (`fleet_prompt.py:190-194`): Keywords to validate AI responses are fleet-related
4. **Context Analysis** (`fleet_prompt.py:197-207`): Contextual terms for better topic detection

## Key Features

- **Topic Validation**: Multi-layer filtering ensures only fleet-related responses
- **Conversation Memory**: Maintains context across chat sessions
- **Sound-Based Diagnostics**: Specialized in identifying vehicle problems through audio descriptions
- **Comprehensive Coverage**: Handles fleet operations, maintenance, driver management, route optimization, compliance, and financials
- **Web Interface**: User-friendly chat interface with example questions and real-time communication

## Development Notes

- The application runs on port 5000 by default
- All responses are validated to ensure they remain fleet-focused
- Conversation history is limited to prevent context overflow
- The DeepSeek model uses temperature=0 for consistent, factual responses
- CORS is enabled for frontend integration
- Docker container includes health checks monitoring API availability and response times
- Uses LangChain for conversation memory management (`langchain-core`, `langchain-community`)
- Includes comprehensive error handling and API response validation

## Dependencies

Key dependencies from `requirements.txt`:
- `Flask==2.3.3` - Web framework
- `langchain==0.1.20` - Conversation memory management
- `requests==2.31.0` - HTTP client for DeepSeek API
- `Flask-CORS==4.0.0` - Cross-origin resource sharing
- `python-dotenv==1.0.0` - Environment variable management

## Cost Analysis

**DeepSeek API Costs:**
- Input tokens: ~$0.14 per 1M tokens
- Output tokens: ~$0.28 per 1M tokens  
- Average cost per request: ~$0.00037

**Monthly estimates:** 100 requests/day = ~$11.10, 1,000 requests/day = ~$111.00