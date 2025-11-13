FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \n    PIP_DISABLE_PIP_VERSION_CHECK=1 \n    PYTHONDONTWRITEBYTECODE=1 \n    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3 python3-pip git ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python3", "inference/generate.py", "--config", "configs/infer.yaml", "--prompt_file", "eval/prompts.txt"]