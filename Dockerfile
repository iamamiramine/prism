FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y curl wget gcc build-essential git gh ffmpeg

# install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
  /bin/bash ~/miniconda.sh -b -p /opt/conda

RUN /opt/conda/bin/conda create -y -n myenv python=3.10

WORKDIR /app
ENV PATH=/opt/conda/envs/myenv/bin:$PATH

RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git clone https://github.com/dzluke/Sound-Diffusion.git

COPY ./src ./src
COPY ./app.py ./app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]