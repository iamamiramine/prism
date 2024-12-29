from pathlib import Path
import time
from omegaconf import OmegaConf
import librosa
from subprocess import run, Popen, PIPE
import threading
import queue
import sounddevice as sd
import numpy as np

import torch
import math

from typing import List, Tuple, Optional, Callable

from src.application.audio_reactive.helpers.generate_helper import text2img, img2img, load_model_from_config
from src.application.audio_reactive.helpers import utils_helper
from src.domain.models.audio_reactive.generate_models import GenerateImageParameters, GenerateVideoParameters, GenerateAudioReactiveVideoParameters

# model from https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt

def generate_image(parameters: GenerateImageParameters) -> Path:
    """
    Generate variations of images using a stable diffusion model with network bending.
    
    Args:
        parameters: GenerateImageParameters
        
    Returns:
        Path to the directory containing generated images
    """
    # Set default layers if none provided
    if parameters.layers is None:
        parameters.layers = [0, 10, 20, 30, 40, 49]

    # initialize paths
    image_storage_path = Path(parameters.image_storage_path)
    image_storage_path.mkdir(exist_ok=True)
    utils_helper.clear_dir(image_storage_path)
    
    tic = time.time()

    # Create args object to maintain compatibility with existing functions
    class Args:
        pass
    args = Args()
    args.skip_grid = parameters.skip_grid
    args.skip_save = parameters.skip_save
    args.ddim_steps = parameters.ddim_steps
    args.plms = parameters.plms
    args.laion400m = parameters.laion400m
    args.fixed_code = parameters.fixed_code
    args.ddim_eta = parameters.ddim_eta
    args.n_iter = parameters.n_iter
    args.H = parameters.height
    args.W = parameters.width
    args.C = parameters.latent_channels
    args.f = parameters.downsampling_factor
    args.n_samples = parameters.n_samples
    args.n_rows = parameters.n_rows
    args.scale = parameters.guidance_scale
    args.seed = parameters.seed
    args.precision = parameters.precision

    # load model
    config = OmegaConf.load(str(parameters.config_path))
    model = load_model_from_config(config, str(parameters.checkpoint_path))

    # Generate scale ranges
    scales = [(i, i + parameters.scale_increment) for i in [
        round(x, 1) for x in [
            parameters.scale_start + j*parameters.scale_increment 
            for j in range(int((parameters.scale_end - parameters.scale_start) / parameters.scale_increment) + 1)
        ]
    ]]

    num_imgs = len(scales)
    print(f">>> Generating {num_imgs} images")

    # generate frames
    for l in parameters.layers:
        folder = image_storage_path / f'layer{l}'
        folder.mkdir(parents=True, exist_ok=True)
        for s in scales:
            args.seed = parameters.seed  # reset seed for each generation
            bend = utils_helper.clamp(s)
            text2img(model, parameters.prompt, folder, args, bend, l)

    print(">>> Generated {} images".format(num_imgs))
    print(">>> Took", utils_helper.time_string(time.time() - tic))
    print(">>> Avg time per img: ", utils_helper.time_string((time.time() - tic) / num_imgs))
    print("Done.")
    
    return image_storage_path


def generate_video(parameters: GenerateVideoParameters) -> Tuple[Path, List[float]]:
    """
    Generate an audio-reactive video using a stable diffusion model.
    
    Args:
        parameters: GenerateVideoParameters
        
    Returns:
        Tuple containing:
        - Path to the generated video file
        - List of stored audio feature combinations
        
    Raises:
        ValueError: If the specified bend_function or audio_feature_function doesn't exist in utils_helper
    """
    # Validate functions exist
    if not hasattr(utils_helper, parameters.bend_function):
        raise ValueError(f"Bend function '{parameters.bend_function}' not found in utils_helper")
    if not hasattr(utils_helper, parameters.audio_feature_function):
        raise ValueError(f"Audio feature function '{parameters.audio_feature_function}' not found in utils_helper")

    # Initialize paths
    image_storage_path = Path(parameters.image_storage_path)
    output_video_path = Path(parameters.output_video_path)
    audio_path = Path(parameters.audio_path)
    
    output_video_path.mkdir(exist_ok=True)
    image_storage_path.mkdir(exist_ok=True)
    utils_helper.clear_dir(image_storage_path)
    
    utils_helper.set_sampling_rate(parameters.sampling_rate)
    tic = time.time()

    # Create args object to maintain compatibility with existing functions
    class Args:
        pass
    args = Args()
    args.skip_grid = parameters.skip_grid
    args.skip_save = parameters.skip_save
    args.ddim_steps = parameters.ddim_steps
    args.plms = parameters.plms
    args.laion400m = parameters.laion400m
    args.fixed_code = parameters.fixed_code
    args.ddim_eta = parameters.ddim_eta
    args.n_iter = parameters.n_iter
    args.H = parameters.height
    args.W = parameters.width
    args.C = parameters.latent_channels
    args.f = parameters.downsampling_factor
    args.n_samples = parameters.n_samples
    args.n_rows = parameters.n_rows
    args.scale = parameters.guidance_scale
    args.seed = parameters.seed
    args.precision = parameters.precision
    args.fps = parameters.fps
    args.audio = str(audio_path)

    # Load model
    config = OmegaConf.load(str(parameters.config_path))
    model = load_model_from_config(config, str(parameters.checkpoint_path))

    # Set up prompts
    prompt2 = parameters.prompt2 if parameters.prompt2 is not None else parameters.prompt1
    prompt = (parameters.prompt1, parameters.prompt2)

    # Load and process audio
    audio, _ = librosa.load(audio_path, sr=parameters.sampling_rate)
    frame_length = 1. / parameters.fps
    num_frames = int(audio.size // (frame_length * parameters.sampling_rate)) + 1

    # Set up batched noise if enabled
    if parameters.do_batched_noise:
        noise = torch.empty((parameters.width // parameters.downsampling_factor, parameters.height // parameters.downsampling_factor, parameters.latent_channels), dtype=torch.float64)
        walk_noise_x = torch.distributions.normal.Normal(0, 1).sample(noise.shape).double()
        walk_noise_y = torch.distributions.normal.Normal(0, 1).sample(noise.shape).double()

        walk_scale_x = torch.cos(torch.linspace(0, 2, num_frames) * math.pi).double()
        walk_scale_y = torch.sin(torch.linspace(0, 2, num_frames) * math.pi).double()
        noise_x = torch.tensordot(walk_scale_x, walk_noise_x, dims=0)
        noise_y = torch.tensordot(walk_scale_y, walk_noise_y, dims=0)
        batched_noise = noise_x + noise_y
    else:
        batched_noise = None

    stored_combos = []
    print(f">>> Generating {num_frames} frames")

    curr_slice, prev_slice = None, None
    # Generate frames
    for i in range(num_frames):
        slice_start = int(i * frame_length * parameters.sampling_rate)
        slice_end = int((i + 1) * frame_length * parameters.sampling_rate)
        audio_slice = audio[slice_start:slice_end]
        curr_slice = utils_helper.spectrum(audio_slice, parameters.sampling_rate)[0]

        # Extract and process audio features
        audio_feature = getattr(utils_helper, parameters.audio_feature_function)(audio_slice, parameters.sampling_rate) / 10.
        combo = audio_feature / parameters.audio_feature_scale
        stored_combos.append(combo)

        print(">>> Audio Feature Value:", combo)
        bend = getattr(utils_helper, parameters.bend_function)(combo)
        
        args.num_frames = num_frames
        args.frame = i
        args.seed = parameters.seed

        # Generate frame
        noise = batched_noise[i].to(device='cuda') if batched_noise is not None else None
        text2img(model, prompt, image_storage_path, args, bend, parameters.layer, noise=noise)
        
        # Create in-progress video every 10 seconds
        if i % (10 * parameters.fps) == 0:
            ffmpeg_command = [
                "ffmpeg", "-y",
                "-framerate", str(parameters.fps),
                "-i", str(image_storage_path) + "/%05d.png",
                "-i", str(audio_path),
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "in_progress.mp4"
            ]
            run(ffmpeg_command)
        
        prev_slice = curr_slice

    # Generate output video name using the string names directly
    video_name = output_video_path / f"{audio_path.stem}_{parameters.bend_function}_{parameters.audio_feature_function}_layer{parameters.layer}_{parameters.fps}fps.mp4"
    counter = 1
    while video_name.exists():
        video_name = output_video_path / f"{audio_path.stem}_{parameters.bend_function}_{parameters.audio_feature_function}_layer{parameters.layer}_{parameters.fps}fps{counter}.mp4"
        counter += 1

    # Create final video
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-framerate", str(parameters.fps),
        "-i", str(image_storage_path) + "/%05d.png",
        "-i", str(audio_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        str(video_name)
    ]
    run(ffmpeg_command)

    print(">>> Generated {} images".format(num_frames))
    print(">>> Took", utils_helper.time_string(time.time() - tic))
    print(">>> Avg time per frame: ", utils_helper.time_string((time.time() - tic) / num_frames))
    print("Done.")

    return video_name, stored_combos



def generate_audio_reactive_video(parameters: GenerateAudioReactiveVideoParameters) -> Path:
    """
    Generate an audio-reactive video using a stable diffusion model.
    
    Args:
        audio_path: Path to the input audio file
        config_path: Path to the model config file
        checkpoint_path: Path to the model checkpoint
        fps: Frames per second for the output video
        prompt: Text prompt for the initial image generation
        layer: Layer number for network bending
        seed: Initial seed for generation
        sampling_rate: Audio sampling rate
        image_storage_path: Path to store intermediate frames
        output_video_path: Path to store the output video
        
    Returns:
        Path to the generated video file
    """
    # initialize paths and utilities
    utils_helper.set_sampling_rate(parameters.sampling_rate)
    output_video_path = Path(parameters.output_video_path)
    output_video_path.mkdir(parents=True, exist_ok=True)
    image_storage_path = Path(parameters.image_storage_path)
    image_storage_path.mkdir(parents=True, exist_ok=True)
    utils_helper.clear_dir(image_storage_path)
    
    tic = time.time()
    
    # ensure paths are Path objects
    audio_path = Path(parameters.audio_path)
    
    # load input audio
    audio, _ = librosa.load(audio_path, sr=parameters.sampling_rate)
    
    # load model
    config = OmegaConf.load(str(parameters.config_path))
    model = load_model_from_config(config, str(parameters.checkpoint_path))
    
    # calculate number of frames needed
    frame_length = 1. / parameters.fps  # length of each frame in seconds
    num_frames = int(audio.size // (frame_length * parameters.sampling_rate)) + 1
    
    print(f">>> Generating {num_frames} frames")
    
    # create a simple args object to maintain compatibility
    class Args:
        pass
    args = Args()
    args.fps = parameters.fps
    args.seed = parameters.seed
    
    # generate the first frame
    encoding = model.get_learned_conditioning(parameters.prompt)
    text2img(model, parameters.prompt, image_storage_path, args)
    
    # generate subsequent frames
    for i in range(1, num_frames):
        slice_start = int(i * frame_length * parameters.sampling_rate)
        slice_end = int((i + 1) * frame_length * parameters.sampling_rate)
        audio_slice = audio[slice_start:slice_end]
        
        # bend is a function that defines how to apply network bending given a latent tensor and audio
        bend = utils_helper.subtract_full
        bend_function_name = bend.__name__
        audio_feature = utils_helper.skewness
        audio_feature_name = audio_feature.__name__
        audio_feature = audio_feature(audio_slice, parameters.sampling_rate) / 10.
        print(">>> Audio Feature Value:", audio_feature)
        bend = bend(audio_feature)
        
        init_img_path = image_storage_path / f"{(i - 1):05}.png"
        args.seed += 1
        
        img2img(model, encoding, init_img_path, image_storage_path, bend, parameters.layer, args)
        
        # every 10 seconds, create an in progress video
        if i % (10 * parameters.fps) == 0:
            ffmpeg_command = ["ffmpeg",
                            "-y",
                            "-framerate", str(parameters.fps),
                            "-i", str(image_storage_path) + "/%05d.png",
                            "-i", str(audio_path),
                            "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p",
                            "in_progress.mp4"]
            run(ffmpeg_command)
    
    # generate output video name
    video_name = output_video_path / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{parameters.layer}_{parameters.fps}fps.mp4"
    counter = 1
    while video_name.exists():
        video_name = output_video_path / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{parameters.layer}_{parameters.fps}fps{counter}.mp4"
        counter += 1
    
    # create final video
    ffmpeg_command = ["ffmpeg",
                     "-y",
                     "-framerate", str(parameters.fps),
                     "-i", str(parameters.image_storage_path) + "/%05d.png",
                     "-i", str(audio_path),
                     "-vcodec", "libx264",
                     "-pix_fmt", "yuv420p",
                     str(video_name)]
    run(ffmpeg_command)
    
    print(">>> Generated {} images".format(num_frames))
    print(">>> Took", utils_helper.time_string(time.time() - tic))
    print(">>> Avg time per frame: ", utils_helper.time_string((time.time() - tic) / num_frames))
    print("Done.")
    
    return video_name


def generate_realtime_video(parameters: GenerateAudioReactiveVideoParameters) -> None:
    """
    Generate an audio-reactive video in real-time using a stable diffusion model.
    This function processes audio input in real-time and generates frames on the fly.
    
    Args:
        parameters: GenerateAudioReactiveVideoParameters
    """
    # Initialize paths and utilities
    utils_helper.set_sampling_rate(parameters.sampling_rate)
    image_storage_path = Path(parameters.image_storage_path)
    image_storage_path.mkdir(parents=True, exist_ok=True)
    utils_helper.clear_dir(image_storage_path)
    
    # Load model
    config = OmegaConf.load(str(parameters.config_path))
    model = load_model_from_config(config, str(parameters.checkpoint_path))
    
    # Create frame queue and processing flags
    frame_queue = queue.Queue(maxsize=30)  # Buffer 1 second of frames at 30fps
    is_processing = True
    
    # Create args object
    class Args:
        pass
    args = Args()
    args.fps = parameters.fps
    args.seed = parameters.seed
    args.ddim_steps = 20  # Reduced steps for faster generation
    args.skip_grid = True
    args.skip_save = False
    args.plms = False
    args.fixed_code = True
    args.ddim_eta = 0.0
    args.n_iter = 1
    args.H = parameters.height if hasattr(parameters, 'height') else 512
    args.W = parameters.width if hasattr(parameters, 'width') else 512
    args.C = parameters.latent_channels if hasattr(parameters, 'latent_channels') else 4
    args.f = parameters.downsampling_factor if hasattr(parameters, 'downsampling_factor') else 8
    args.scale = 7.5
    
    # Initialize ffmpeg process for streaming
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'image2pipe',
        '-vcodec', 'png',
        '-r', str(parameters.fps),
        '-i', '-',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'mpegts',
        '-tune', 'zerolatency',
        'http://localhost:7000/live.stream'
    ]
    ffmpeg_process = Popen(ffmpeg_cmd, stdin=PIPE)
    
    def audio_callback(indata, frames, time, status):
        """Callback for processing audio chunks"""
        if status:
            print(f"Status: {status}")
        
        # Process audio chunk
        audio_chunk = indata[:, 0]  # Take first channel if stereo
        audio_feature = utils_helper.skewness(audio_chunk, parameters.sampling_rate) / 10.
        bend = utils_helper.subtract_full(audio_feature)
        
        # Put in queue for frame generation
        if not frame_queue.full():
            frame_queue.put((audio_feature, bend))
    
    def frame_generator():
        """Thread function for generating frames"""
        # Generate initial frame
        encoding = model.get_learned_conditioning(parameters.prompt)
        prev_frame = None
        
        while is_processing:
            try:
                audio_feature, bend = frame_queue.get(timeout=1.0)
                
                if prev_frame is None:
                    # Generate first frame
                    text2img(model, parameters.prompt, image_storage_path, args)
                    prev_frame = image_storage_path / "00000.png"
                else:
                    # Generate subsequent frame
                    img2img(model, encoding, prev_frame, image_storage_path, bend, parameters.layer, args)
                    prev_frame = image_storage_path / "00001.png"
                
                # Stream frame to ffmpeg
                with open(prev_frame, 'rb') as f:
                    ffmpeg_process.stdin.write(f.read())
                    ffmpeg_process.stdin.flush()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame generation: {e}")
                break
    
    # Start frame generation thread
    frame_thread = threading.Thread(target=frame_generator)
    frame_thread.start()
    
    try:
        # Start real-time audio processing
        with sd.InputStream(channels=1,
                          samplerate=parameters.sampling_rate,
                          blocksize=int(parameters.sampling_rate / parameters.fps),
                          callback=audio_callback):
            print("Starting real-time generation. Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping real-time generation...")
    finally:
        # Cleanup
        is_processing = False
        frame_thread.join()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

