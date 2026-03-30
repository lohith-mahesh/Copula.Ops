FROM python:3.11-slim
WORKDIR /code
ENV PYTHONUNBUFFERED=1 
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN chmod -R 777 /code
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
