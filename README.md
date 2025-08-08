# Fleet Agent API

A specialized AI-powered Fleet Management API that provides expert guidance on fleet operations, vehicle management, driver coordination, and logistics topics.

## Features

- 🚛 **Fleet Operations**: Vehicle acquisition, deployment, utilization, fleet sizing
- 👨‍💼 **Driver Management**: Recruitment, training, scheduling, performance, safety
- 🛣️ **Route Optimization**: Planning, dispatching, load optimization, delivery scheduling
- 🔧 **Maintenance & Repairs**: Preventive maintenance, emergency repairs, parts management
- 🚗 **Vehicle Diagnostics**: Engine, transmission, brake, steering, suspension troubleshooting
- 🔊 **Sound-Based Diagnosis**: Identifying problems through unusual noises and sounds
- ⛽ **Fuel Management**: Consumption tracking, cost optimization, efficiency improvements
- 📱 **Fleet Technology**: Telematics, GPS tracking, fleet software, ELD compliance
- 📋 **Compliance**: DOT regulations, licensing, inspections, safety standards
- 💰 **Fleet Financials**: TCO, budgeting, lease vs buy, cost analysis, ROI

## API Endpoints

- `POST /chat` - Send a question to the fleet agent
- `GET /health` - Check API health status
- `GET /` - API information and usage examples

## Docker Deployment

### Prerequisites

- Docker and Docker Compose installed
- DeepSeek API key

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd v1-chatbot
   ```

2. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env file with your DeepSeek API key
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Test the API**
   ```bash
   curl -X POST http://localhost:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I optimize fuel consumption for my fleet?"}'
   ```

### Manual Docker Build

```bash
# Build the image
docker build -t fleet-agent-api .

# Run the container
docker run -p 5000:5000 \
  -e DEEPSEEK_API_KEY=your_api_key_here \
  fleet-agent-api
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEEPSEEK_API_KEY` | Your DeepSeek API key | Yes |
| `FLASK_ENV` | Flask environment (production/development) | No |
| `FLASK_DEBUG` | Enable Flask debug mode | No |

## API Usage Examples

### Fleet Management Question
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the best practices for preventive maintenance scheduling?"
  }'
```

### Vehicle Diagnostics
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "My truck is making a grinding sound when braking. What could be wrong?"
  }'
```

### Route Optimization
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How can I optimize delivery routes for my 10-vehicle fleet?"
  }'
```

## Cost Analysis

### DeepSeek API Costs
- **Input tokens**: ~$0.14 per 1M tokens
- **Output tokens**: ~$0.28 per 1M tokens
- **Average cost per request**: ~$0.00037

### Monthly Cost Estimates
| Requests/Day | Monthly Cost |
|-------------|--------------|
| 100         | ~$11.10     |
| 1,000       | ~$111.00    |
| 10,000      | ~$1,110.00  |

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export DEEPSEEK_API_KEY=your_api_key_here

# Run the application
python fleet_agent_api.py
```

### Testing
```bash
# Health check
curl http://localhost:5000/health

# API info
curl http://localhost:5000/
```

## Security Features

- Non-root user in Docker container
- Environment variable configuration
- Input validation and sanitization
- Fleet-specific topic filtering
- Response validation

## Monitoring

The Docker container includes health checks that monitor:
- API endpoint availability
- Response times
- Service status

## Troubleshooting

### Common Issues

1. **API Key Not Configured**
   - Ensure `DEEPSEEK_API_KEY` is set in your environment
   - Check the `.env` file exists and contains the key

2. **Port Already in Use**
   - Change the port mapping in `docker-compose.yml`
   - Or stop other services using port 5000

3. **Container Won't Start**
   - Check Docker logs: `docker-compose logs`
   - Verify all required files are present

## License

This project is licensed under the MIT License. 