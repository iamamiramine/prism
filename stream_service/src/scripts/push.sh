#!/bin/bash

while true; do
	ffmpeg -re -i ../shared/video/latent_walk.mp4 -c:v copy -f rtsp rtsp://127.0.0.1:5554/stream1
done
