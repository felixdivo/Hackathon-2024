# syntax=docker/dockerfile:1

FROM python:3.12

RUN mkdir /app
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "evaluate/eval.py"]
