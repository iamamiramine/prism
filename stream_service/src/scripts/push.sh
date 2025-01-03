#!/bin/bash

while true; do
	ffmpeg -re -i ../shared/video/latent_walk.mp4 \
	-c:v copy \
	-f rtp_mpegts \
	rtp://127.0.0.1:10000 \
	-sdp_file stream.sdp
	
	sleep 1
done
