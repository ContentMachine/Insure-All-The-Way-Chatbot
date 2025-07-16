# Use an official Python image
FROM python:3.10

# Set work directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Start server
CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "8000"]
