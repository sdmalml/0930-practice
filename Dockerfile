FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r  requirements.txt

COPY *.py .

RUN python train.py

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0"]