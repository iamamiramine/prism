FROM ubuntu:22.04

RUN apt-get update && apt-get install -y wget curl ffmpeg \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base  \
    gstreamer1.0-plugins-good  \
    gstreamer1.0-plugins-bad  \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav  \
    gstreamer1.0-tools  \
    gstreamer1.0-x  \
    gstreamer1.0-alsa  \
    gstreamer1.0-gl  \
    gstreamer1.0-gtk3  \
    gstreamer1.0-qt5  \
    gstreamer1.0-pulseaudio \
    gir1.2-gst-plugins-bad-1.0  \
    gstreamer1.0-nice \
    python3  \
    python3-pip \
    npm

WORKDIR /app/stream_service

# Copy application files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

#RUN pip3 install meson
#RUN pip install pygobject

COPY ./src ./

RUN cd scripts && npm install

RUN cd ..

#CMD ["sleep", "infinity"]
CMD ["node", "scripts/index.js"]