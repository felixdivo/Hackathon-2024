# syntax=docker/dockerfile:1

# Use the official Python image
FROM python:3.12

# Install OpenCV and its dependencies
RUN apt-get update && apt-get -yq install python3-opencv

# Add the current directory (the solution) to the container
RUN mkdir /app
WORKDIR /app
COPY . .

# Install the Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Run the evaluation script if nothing else is specified
CMD ["python", "evaluate/eval.py"]
