# Use Python 3.11 slim image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY fleet_agent_api.py .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "main.py"]
