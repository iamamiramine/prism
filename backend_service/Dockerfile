#FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM ubuntu:22.04

# Update
RUN apt-get update

# Install required packages
RUN apt-get install -y curl wget gcc build-essential git gh ffmpeg libportaudio2

# install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
  /bin/bash ~/miniconda.sh -b -p /opt/conda

RUN /opt/conda/bin/conda create -y -n myenv python=3.10

WORKDIR /app/backend_service
ENV PATH=/opt/conda/envs/myenv/bin:$PATH

# Install PyTorch first
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Clone required repositories first
RUN git clone https://github.com/dzluke/Sound-Diffusion.git
RUN git clone https://github.com/CompVis/taming-transformers.git
RUN git clone https://github.com/openai/CLIP.git

# Install taming-transformers and CLIP from local clones
RUN cd taming-transformers && pip install -e .
RUN cd CLIP && pip install -e .

# Copy application files
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ldm ./ldm
COPY ./src ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]