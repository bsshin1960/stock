FROM python:3.11-slim

# Install system dependencies required for Flet and Web requests
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the default port (Hugging Face Spaces automatically forwards 7860)
EXPOSE 7860

# Run the app
CMD ["python", "main.py"]
