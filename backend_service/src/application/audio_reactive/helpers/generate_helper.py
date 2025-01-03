# Standard library imports
from contextlib import nullcontext
from itertools import islice
from einops import rearrange, repeat
import os

# Third-party imports
import cv2
import numpy as np
import torch
from imwatermark import WatermarkEncoder
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import trange
from torch import autocast

# Local imports
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def get_device():
    """Get the most efficient available device with memory optimization settings."""
    if torch.cuda.is_available():
        # # Enable TF32 for better performance on Ampere GPUs
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 = True
        # # Enable cudnn benchmarking for better performance
        # torch.backends.cudnn.benchmark = True
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(get_device())
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    w, h = 256, 256
    print(f"Resizing input image to ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def text2img(model, prompt, outpath, opt, fn=None, layer=None, titles=None, noise=None):
    """
    Generate an image using text2img with optimized performance.
    """
    seed_everything(opt.seed)
    device = torch.device(get_device())

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    batch_size = opt.n_samples
    
    # Optimize start code generation
    if opt.fixed_code:
        start_code = torch.randn(
            [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f],
            device=device,  # Generate directly on device
            dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
    else:
        start_code = None

    # Optimize precision based on device
    precision_scope = nullcontext if opt.precision != "autocast" else torch.cuda.amp.autocast
        
    with torch.no_grad():
        with precision_scope():
            with model.ema_scope():
                # Optimize conditioning
                if isinstance(prompt, str):
                    c = model.get_learned_conditioning(prompt).to(device)
                else:
                    # Optimize prompt interpolation
                    p1 = model.get_learned_conditioning(prompt[0]).to(device)
                    p2 = model.get_learned_conditioning(prompt[1]).to(device)
                    c_linspace = torch.linspace(0, 1, opt.num_frames, device=device)
                    c = torch.lerp(p1, p2, c_linspace[opt.frame])
                
                # Pre-compute unconditional guidance
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                # Optimize sampling
                samples_ddim, _ = sampler.sample(
                    fn, layer,
                    S=opt.ddim_steps,
                    conditioning=c,
                    batch_size=opt.n_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc,
                    eta=opt.ddim_eta,
                    x_T=start_code,
                    noise=noise
                )

                # Optimize first stage decoding
                x_samples = model.decode_first_stage(samples_ddim)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                
                return x_samples[0].to(device)


def img2img(model, encoding, init_image, outpath, fn, opt, layer=None, noise=None):
    """
    Generate an image using img2img with optimized performance.
    """
    strength = opt.strength
    seed_everything(opt.seed)
    device = torch.device(get_device())

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    batch_size = opt.n_samples

    # Handle tensor input with optimized memory usage
    if isinstance(init_image, torch.Tensor):
        if init_image.dim() == 3:
            init_image = init_image.unsqueeze(0)
        # Move to device and match model's precision
        init_image = init_image.to(device).to(model.dtype)
    else:
        assert os.path.isfile(init_image)
        init_image = load_img(init_image).to(device).to(model.dtype)

    assert init_image.dim() == 4, f"Expected 4D tensor [batch, channels, height, width], got shape {init_image.shape}"
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

    # Optimize precision based on device  
    precision_scope = nullcontext if opt.precision != "autocast" else torch.cuda.amp.autocast

    # Pre-encode first stage with optimized memory usage
    with precision_scope():
        # Ensure model and input are in same precision
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= strength < 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * opt.ddim_steps)
      
    with torch.no_grad():
        with precision_scope():
            with model.ema_scope():
                # Pre-compute unconditional guidance
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                
                # Optimize encoding
                z_enc = sampler.stochastic_encode(
                    init_latent,
                    torch.tensor([t_enc] * batch_size, device=device),
                )

                # Optimize decoding
                samples = sampler.decode(
                    fn, layer, z_enc, encoding, t_enc,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc,
                    noise=noise
                )

                # Optimize first stage decoding
                with torch.cuda.amp.autocast() if device.type == 'cuda' else nullcontext():
                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                
                return x_samples[0].to(device)
