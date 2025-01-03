# Standard library imports
import time
import os
from pathlib import Path
from subprocess import run
from PIL import Image
import pickle
import numpy as np
from einops import rearrange

# Third-party imports
import torch
import librosa
from omegaconf import OmegaConf

# Local imports
from application.audio_reactive.helpers.generate_helper import text2img, img2img, load_model_from_config, get_device
from application.audio_reactive.helpers import utils_helper
from application.audio_reactive.helpers.utils_helper import subtract_full, skewness
from domain.models.audio_reactive.generate_models import GenerateAudioReactiveVideoParameters

# Global variables
_model = None
_config = None
_CACHE_DIR = Path("../models/ldm/stable_diffusion_v1/cache")
_CACHE_FILE = _CACHE_DIR / "model_cache.pt"

def initialize_model(
        config_path: str = "../shared/configs/stable_diffusion/v1-inference.yaml",
        checkpoint_path: str = "../models/ldm/stable_diffusion_v1/model.ckpt"
    ) -> dict:
    """
    Initialize the model and config globally. Uses caching to speed up loading.
    
    Args:
        config_path: Path to the model config file
        checkpoint_path: Path to the model checkpoint file
    """
    global _model, _config
    
    if _model is not None:
        return {"message": "Model already initialized"}

    device = torch.device(get_device())
    
    # Create cache directory if it doesn't exist
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Try loading from cache first
    if _CACHE_FILE.exists():
        try:
            print("Loading model from cache...")
            _model = torch.load(_CACHE_FILE, map_location=device)
            _model.eval()
            return {"message": "Model loaded from cache successfully"}
        except Exception as e:
            print(f"Failed to load from cache: {e}")
            _CACHE_FILE.unlink() if _CACHE_FILE.exists() else None

    # Load from checkpoint if cache fails or doesn't exist
    try:
        print("Loading model from checkpoint...")
        _config = OmegaConf.load(config_path)
        _model = load_model_from_config(_config, checkpoint_path)
        _model.eval()
        
        # Cache the model
        try:
            torch.save(_model, _CACHE_FILE)
        except Exception as e:
            print(f"Failed to cache model: {e}")
            _CACHE_FILE.unlink() if _CACHE_FILE.exists() else None
        
        return {"message": "Model initialized and cached successfully"}
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {e}")

def get_model():
    """
    Get the initialized model. Raises an error if model is not initialized.
    
    Returns:
        The initialized model
    """
    if _model is None:
        raise RuntimeError("Model not initialized. Call initialize_model first.")
    return _model

# model from https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt

def generate_audio_reactive_video(parameters: GenerateAudioReactiveVideoParameters) -> Path:
    """
    Generate an audio-reactive video using a stable diffusion model.
    """
    # Initialize paths and utilities
    output_video_path = Path(parameters.output_video_path)
    image_storage_path = Path(parameters.image_storage_path)
    audio_path = Path(parameters.audio_path)
    
    # Create directories for final output
    output_video_path.mkdir(parents=True, exist_ok=True)
    image_storage_path.mkdir(parents=True, exist_ok=True)

    tic = time.time()

    # Load audio and calculate frames
    audio, _ = librosa.load(audio_path, sr=parameters.sampling_rate)
    frame_length = 1. / parameters.fps
    num_frames = int(audio.size // (frame_length * parameters.sampling_rate)) + 1

    print(f">>> Generating {num_frames} frames")

    # Get model and prepare args
    model = get_model()
    args = _prepare_args(parameters)
    
    # Pre-compute model conditioning
    encoding = model.get_learned_conditioning(parameters.prompt)
    
    # Generate first frame and keep it in memory
    current_frame = text2img(model, parameters.prompt, image_storage_path, args)
    
    # Store frames in memory
    frames = [current_frame]

    # Pre-compute audio features for all frames
    audio_features = []
    for i in range(1, num_frames):
        slice_start = int(i * frame_length * parameters.sampling_rate)
        slice_end = int((i + 1) * frame_length * parameters.sampling_rate)
        audio_slice = audio[slice_start:slice_end]
        feature = utils_helper.skewness(audio_slice, parameters.sampling_rate) / 10.
        audio_features.append(feature)

    # Generate subsequent frames
    for i, audio_feature in enumerate(audio_features, start=1):
        # Apply network bending
        bend = utils_helper.subtract_full(audio_feature)
        
        args.seed += 1
        args.frame += 1

        # Generate next frame using the current frame
        current_frame = img2img(model, encoding, current_frame, image_storage_path, bend, args, parameters.layer)
        frames.append(current_frame)

        # Create progress video every 10 seconds
        if i % (10 * parameters.fps) == 0:
            _save_frames_and_create_video(frames[:i+1], "in_progress.mp4", image_storage_path, audio_path, parameters.fps)

    # Generate final video with unique name
    video_name = _generate_unique_video_name(
        output_video_path, 
        audio_path.stem, 
        utils_helper.subtract_full.__name__, 
        utils_helper.skewness.__name__, 
        parameters.layer, 
        parameters.fps
    )
    
    # Save all frames and create final video
    _save_frames_and_create_video(frames, str(video_name), image_storage_path, audio_path, parameters.fps)

    print(f">>> Generated {num_frames} images")
    total_time = time.time() - tic
    print(">>> Took", utils_helper.time_string(total_time))
    print(">>> Avg time per frame: ", utils_helper.time_string(total_time / num_frames))

    return video_name

def _save_frames_and_create_video(frames, output_path, image_storage_path, audio_path, fps):
    """Helper function to save frames temporarily and create a video."""
    utils_helper.clear_dir(image_storage_path)
    
    # Temporarily save frames for ffmpeg
    for i, frame in enumerate(frames):
        # Convert tensor to image
        frame_np = 255. * rearrange(frame.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(frame_np.astype(np.uint8))
        img.save(os.path.join(image_storage_path, f"{i:05}.png"))
    
    # Create video
    _create_video(output_path, image_storage_path, audio_path, fps)

def _prepare_args(parameters):
    """Helper function to prepare args object from parameters."""
    args = type('Args', (), {})()
    for key, value in parameters.__dict__.items():
        setattr(args, key, value)
    args.frame = 0
    return args

def _create_video(output_path, image_path, audio_path, fps):
    """Helper function to create video from frames."""
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(image_path) + "/%05d.png",
        "-i", str(audio_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    run(ffmpeg_command)

def _generate_unique_video_name(output_path, stem, bend_name, feature_name, layer, fps):
    """Helper function to generate unique video name."""
    base_name = f"{stem}_{bend_name}_{feature_name}_layer{layer}_{fps}fps.mp4"
    video_name = output_path / base_name
    counter = 1
    while video_name.exists():
        video_name = output_path / f"{stem}_{bend_name}_{feature_name}_layer{layer}_{fps}fps{counter}.mp4"
        counter += 1
    return video_name
