# Training/eval image for Vertex AI custom jobs. Holds only the dependencies; the code is shipped as a
# tarball through the bucket and unpacked at job start (see launch_vertex.sh), so editing code does not
# need a rebuild. Rebuild only when requirements.txt changes:  bash scripts/cloud/launch_vertex.sh build
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends unzip && rm -rf /var/lib/apt/lists/*

# torch is already in the base image; pip keeps it because it satisfies requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/tmp/hf TOKENIZERS_PARALLELISM=false
