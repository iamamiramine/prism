from fastapi import Query

from typing import Annotated, Tuple, Optional, List

from domain.models.base_model import BaseEnum
from pydantic import BaseModel
class GenerateAudioReactiveVideoParameters(BaseModel):
    audio_path: Annotated[str, Query(description="Audio Path")]
    image_storage_path: Annotated[str, Query(description="Image Storage Path")] = "../outputs/image_outputs"
    output_video_path: Annotated[str, Query(description="Output Video Path")] = "../outputs/video_outputs"
    H: Annotated[int, Query(description="Height")] = 512
    W: Annotated[int, Query(description="Width")] = 512
    seed: Annotated[int, Query(description="Seed")] = 46
    bend_function: Annotated[str, Query(description="Name of bend function from utils_helper (e.g., 'rotate_x', 'add_full', etc.)")] = "rotate_x"
    audio_feature_function: Annotated[str, Query(description="Name of audio feature function from utils_helper (e.g., 'centroid', 'spread', etc.)")] = "centroid"
    audio_feature_scale: Annotated[float, Query(description="Audio Feature Scale")] = 250.0
    strength: Annotated[float, Query(description="Strength for img2img")] = 0.75
    fps: Annotated[int, Query(description="FPS")] = 20
    layer: Annotated[int, Query(description="Layer")] = 0
    # Model generation parameters
    skip_grid: Annotated[bool, Query(description="Skip Grid")] = True
    skip_save: Annotated[bool, Query(description="Skip Save")] = False
    ddim_steps: Annotated[int, Query(description="DDIM Steps")] = 50
    plms: Annotated[bool, Query(description="PLMS")] = False
    laion400m: Annotated[bool, Query(description="Laion400m")] = False
    fixed_code: Annotated[bool, Query(description="Fixed Code")] = False
    ddim_eta: Annotated[float, Query(description="DDIM Eta")] = 0.0
    n_iter: Annotated[int, Query(description="N Iter")] = 1
    C: Annotated[int, Query(description="Latent Channels")] = 4
    f: Annotated[int, Query(description="Downsampling Factor")] = 8
    n_samples: Annotated[int, Query(description="N Samples")] = 1
    n_rows: Annotated[int, Query(description="N Rows")] = 0
    scale: Annotated[float, Query(description="Guidance Scale")] = 7.5
    precision: Annotated[str, Query(description="Precision")] = "autocast"
    prompt: Annotated[str, Query(description="Prompt")] = "a floating orb"
    sampling_rate: Annotated[int, Query(description="Sampling Rate")] = 4410
    do_batched_noise: Annotated[bool, Query(description="Do Batched Noise")] = True
    num_frames: Annotated[int, Query(description="Number of Frames")] = 300


