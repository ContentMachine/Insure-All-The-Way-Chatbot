FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install numpy==1.26.4 faiss-cpu && pip install -r requirements.txt


CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "8000"]
