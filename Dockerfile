FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directory for ChromaDB persistence
RUN mkdir -p /app/chroma_db

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the main tutorial script
CMD ["python", "main.py"]
