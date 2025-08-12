# Fleet Agent API Documentation

## Overview
Fleet Agent API provides endpoints for interacting with your fleet agent chatbot. The API is implemented in Python and can be run using Docker or directly with Python.

## Base URL
```
http://localhost:8000/
```
*(Adjust according to your deployment)*

---

## Endpoints

### 1. `/chat`
**POST**  
Initiate a chat with the fleet agent.

**Request Body:**
```json
{
  "message": "Your message here"
}
```

**Response:**
```json
{
  "response": "Agent's reply"
}
```

---

### 2. `/status`
**GET**  
Check the status of the fleet agent.

**Response:**
```json
{
  "status": "online",
  "details": "Agent is running"
}
```

---

### 3. `/config`
**GET**  
Retrieve current configuration of the fleet agent.

**Response:**
```json
{
  "config": {
    // configuration details
  }
}
```

---

## Error Handling

- **400 Bad Request**: Invalid input data.
- **500 Internal Server Error**: Server-side error.

---

## Running the API

### Using Docker
```powershell
docker-compose up
```

### Using Python
```powershell
python fleet_agent_api.py
```

---

## Authentication

Currently, the API does not require authentication. For production, consider adding authentication and authorization.

---

## Contact

For support, refer to the `README.md` or contact the maintainer.

---
