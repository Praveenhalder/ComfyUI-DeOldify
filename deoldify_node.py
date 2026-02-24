"""
DeOldify ComfyUI Custom Node
Integrates DeOldify colorization into ComfyUI pipelines.

Setup:
  1. Clone DeOldify into ComfyUI/custom_nodes/deoldify_comfyui_node/deoldify/
  2. Place weights in ComfyUI/models/deoldify/models/
       ColorizeArtistic_gen.pth
       ColorizeStable_gen.pth
     — OR — let the node auto-download them from HuggingFace on first use.
  3. pip install fastai==1.0.61 pandas opencv-python IPython scipy scikit-image matplotlib bottleneck numexpr
"""

import sys
import shutil
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image as PilImage, ImageEnhance
import comfy.model_management
from huggingface_hub import hf_hub_download

import folder_paths  # ComfyUI's path registry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

NODE_DIR = Path(__file__).parent.resolve()
DEOLDIFY_DIR = NODE_DIR / "deoldify"

# fastai resolves: root_folder / "models" / weights_name + ".pth"
DEOLDIFY_MODELS_DIR = Path(folder_paths.models_dir) / "deoldify"
DEOLDIFY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
(DEOLDIFY_MODELS_DIR / "models").mkdir(parents=True, exist_ok=True)

# HuggingFace repo that hosts the DeOldify weights
HF_REPO_ID = "spensercai/DeOldify"

if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

# ---------------------------------------------------------------------------
# PyTorch 2.6 compat — fastai checkpoints embed arbitrary classes
# ---------------------------------------------------------------------------

def _patch_torch_load():
    import torch as _t
    if getattr(_t.load, "_deoldify_patched", False):
        return
    _orig = _t.load

    def _patched(f, map_location=None, pickle_module=None, **kw):
        kw["weights_only"] = False
        if pickle_module is not None:
            return _orig(f, map_location=map_location, pickle_module=pickle_module, **kw)
        return _orig(f, map_location=map_location, **kw)

    _patched._deoldify_patched = True
    _t.load = _patched
    logging.info("[DeOldify] Patched torch.load → weights_only=False")

_patch_torch_load()

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _ok(name):
    try: __import__(name); return True
    except ImportError: return False

def _check_dependencies():
    REQUIRED = [
        ("fastai==1.0.61", "fastai"), ("pandas", "pandas"),
        ("opencv-python",  "cv2"),    ("IPython", "IPython"),
        ("scipy",          "scipy"),  ("scikit-image", "skimage"),
        ("matplotlib",     "matplotlib"), ("bottleneck", "bottleneck"),
        ("numexpr",        "numexpr"),
    ]
    missing = [pip for pip, imp in REQUIRED if not _ok(imp)]
    if missing:
        cmd = " ".join(missing)
        raise ImportError(
            f"Missing packages: {', '.join(missing)}\n\n"
            f"ComfyUI portable (Windows, run from ComfyUI root):\n"
            f"  .\\python_embeds\\python.exe -m pip install {cmd}\n\n"
            f"Standard Python / venv:\n"
            f"  pip install {cmd}"
        )

def _import_deoldify():
    _check_dependencies()
    try:
        from deoldify import device as device_settings
        from deoldify.device_id import DeviceId
        from deoldify.generators import gen_inference_wide, gen_inference_deep
        from deoldify.filters import ColorizerFilter, MasterFilter
        return device_settings, DeviceId, gen_inference_wide, gen_inference_deep, ColorizerFilter, MasterFilter
    except ImportError as e:
        raise ImportError(
            f"Could not import DeOldify. Clone it into:\n  {DEOLDIFY_DIR}\n"
            f"  git clone https://github.com/jantic/DeOldify \"{DEOLDIFY_DIR}\"\n"
            f"Original error: {e}"
        )

# ---------------------------------------------------------------------------
# Auto-download weights from HuggingFace
# ---------------------------------------------------------------------------

# Map model_type → filename expected inside the HF snapshot
_HF_WEIGHT_FILES = {
    "stable":   "ColorizeStable_gen.pth",
    "video":   "ColorizeVideo_gen.pth",
    
}


def _download_weights_from_hf(weights_name: str, dest_path: Path) -> None:
    """
    Download a single DeOldify weight file from HuggingFace using hf_hub_download
    (downloads only the requested .pth, not the entire repo) then copies it to dest_path.
    """
    logging.info(
        f"[DeOldify] Weights not found at {dest_path}. "
        f"Downloading '{weights_name}' from HuggingFace repo '{HF_REPO_ID}' …"
    )

    # Notify ComfyUI that we're about to do a potentially slow download
    try:
        comfy.model_management.load_models_gpu([])  # no-op, but touches the subsystem
    except Exception:
        pass

    print(
        f"\n[DeOldify] ⬇  Downloading {weights_name} from HuggingFace: {HF_REPO_ID}/{weights_name}\n"
        f"           Destination: {dest_path}\n"
        f"           This may take a few minutes on first run …\n"
    )

    try:
        cached_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=weights_name,
        )
    except Exception as e:
        raise RuntimeError(
            f"[DeOldify] Failed to download '{weights_name}' from HuggingFace repo '{HF_REPO_ID}'.\n"
            f"  Make sure you have internet access and 'huggingface_hub' installed.\n"
            f"  Original error: {e}"
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_file, dest_path)
    logging.info(f"[DeOldify] Weights saved → {dest_path}")
    print(f"[DeOldify] ✓  Weights ready: {dest_path}\n")


def _ensure_weights(weights_name: str) -> Path:
    """
    Return the Path to a ready-to-use .pth file, downloading from HuggingFace
    automatically if it is missing.

    Resolution order:
      1. DEOLDIFY_MODELS_DIR/models/<weights_name>.pth  (canonical fastai path)
      2. DEOLDIFY_MODELS_DIR/<weights_name>.pth          (flat layout — moved to canonical)
      3. Auto-download from HuggingFace → canonical path
    """
    nested_path = DEOLDIFY_MODELS_DIR / "models" / f"{weights_name}.pth"
    flat_path   = DEOLDIFY_MODELS_DIR / f"{weights_name}.pth"

    if nested_path.exists():
        return nested_path

    if flat_path.exists():
        flat_path.rename(nested_path)
        logging.info(f"[DeOldify] Moved {flat_path.name} → models/")
        return nested_path

    # Neither location has the file — auto-download
    _download_weights_from_hf(f"{weights_name}.pth", nested_path)

    if not nested_path.exists():
        raise FileNotFoundError(
            f"[DeOldify] Download appeared to succeed but file is still missing: {nested_path}"
        )
    return nested_path


# ---------------------------------------------------------------------------
# Infer nf_factor from the checkpoint itself
# ---------------------------------------------------------------------------

def _infer_nf_factor(weights_path: Path) -> int:
    """
    Read the saved state_dict and derive nf_factor from the first UnetBlock
    pixel-shuffle conv weight shape, without instantiating any model.

    DynamicUnetWide sets:  nf = 512 * nf_factor
    UnetBlockWide.shuf:   CustomPixelShuffle_ICNR(up_in_c, nf//2, ...)
                           conv weight shape: (nf//2 * scale**2, up_in_c, 1, 1)
                                            = (nf//2 * 4,        up_in_c, 1, 1)

    For the first decoder block (layers.4), up_in_c == ni (encoder bottleneck,
    always 2048 for resnet101).  So:
        weight shape[0] = nf//2 * 4 = 512 * nf_factor * 2
        nf_factor = weight_shape[0] / (512 * 2) = weight_shape[0] / 1024

    Returns the inferred integer nf_factor (1 or 2) or 2 as a safe default.
    """
    try:
        ckpt = torch.load(str(weights_path), map_location="cpu")
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

        key = "layers.4.shuf.conv.0.weight_orig"
        if key in sd:
            out_channels = sd[key].shape[0]
            nf_factor = round(out_channels / 1024)
            logging.info(f"[DeOldify] Inferred nf_factor={nf_factor} from checkpoint "
                         f"(key={key}, shape[0]={out_channels})")
            return max(1, nf_factor)
    except Exception as e:
        logging.warning(f"[DeOldify] Could not infer nf_factor from checkpoint: {e}")
    logging.info("[DeOldify] Falling back to nf_factor=2")
    return 2


# ---------------------------------------------------------------------------
# Colorizer cache
# ---------------------------------------------------------------------------

_colorizer_cache: dict = {}


def _get_colorizer(model_type: str, render_factor: int, device: str):
    cache_key = (model_type, device)
    if cache_key in _colorizer_cache:
        return _colorizer_cache[cache_key]

    device_settings, DeviceId, gen_inference_wide, gen_inference_deep, ColorizerFilter, MasterFilter = _import_deoldify()

    if device == "cuda" and torch.cuda.is_available():
        device_settings.set(DeviceId.GPU0)
    else:
        device_settings.set(DeviceId.CPU)

    weights_map = {
        "stable":   "ColorizeStable_gen",
        "video":   "ColorizeVideo_gen",
    }
    weights_name = weights_map[model_type]

    # Ensure weights exist, downloading from HuggingFace if needed
    nested_path = _ensure_weights(weights_name)

    # Derive nf_factor by inspecting the checkpoint — avoids all guesswork
    nf_factor = _infer_nf_factor(nested_path)

    if model_type == "artistic":
        learn = gen_inference_wide(
            root_folder=DEOLDIFY_MODELS_DIR,
            weights_name=weights_name,
            nf_factor=nf_factor,
        )
        logging.info(f"[DeOldify] Loaded {weights_name} — arch=wide, nf_factor={nf_factor}")
    else:
        # stable: try deep first (official), fall back to wide if shapes mismatch
        ckpt = torch.load(str(nested_path), map_location="cpu")
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        is_deep = any("layers.4.conv1" in k for k in sd.keys())

        if is_deep:
            learn = gen_inference_deep(
                root_folder=DEOLDIFY_MODELS_DIR,
                weights_name=weights_name,
                nf_factor=float(nf_factor),
            )
            logging.info(f"[DeOldify] Loaded {weights_name} — arch=deep, nf_factor={nf_factor}")
        else:
            learn = gen_inference_wide(
                root_folder=DEOLDIFY_MODELS_DIR,
                weights_name=weights_name,
                nf_factor=nf_factor,
            )
            logging.info(f"[DeOldify] Loaded {weights_name} — arch=wide, nf_factor={nf_factor}")

    colorizer = MasterFilter(
        filters=[ColorizerFilter(learn=learn)],
        render_factor=render_factor,
    )
    _colorizer_cache[cache_key] = colorizer
    return colorizer


# ---------------------------------------------------------------------------
# Tensor <-> PIL  (ComfyUI: BHWC float32 [0,1])
# ---------------------------------------------------------------------------

def _tensor_to_pil(tensor: torch.Tensor) -> PilImage.Image:
    np_img = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return PilImage.fromarray(np_img, mode="RGB")

def _pil_to_tensor(image: PilImage.Image) -> torch.Tensor:
    np_img = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(np_img)

def _adjust_saturation(image: PilImage.Image, factor: float) -> PilImage.Image:
    """
    Adjust the color saturation of a PIL image.

    factor = 0.0  → fully desaturated (grayscale)
    factor = 1.0  → original saturation (no change)
    factor > 1.0  → boosted saturation
    """
    if factor == 1.0:
        return image
    return ImageEnhance.Color(image).enhance(factor)


# ---------------------------------------------------------------------------
# ComfyUI Nodes
# ---------------------------------------------------------------------------

class DeOldifyColorize:
    """Colorizes a grayscale or faded image using DeOldify."""

    CATEGORY = "image/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colorize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": (["stable", "video" ], {"default": "stable"}),
                "render_factor": ("INT", {"default": 35, "min": 8, "max": 45, "step": 1}),
                "post_process": ("BOOLEAN", {"default": True}),
                "saturation": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    def colorize(self, image, model_type="stable", render_factor=35,
                 post_process=True, saturation=1.0, device="cuda"):
        colorizer = _get_colorizer(model_type, render_factor, device)
        frames = []
        for i in range(image.shape[0]):
            pil_in = _tensor_to_pil(image[i])
            pil_out = colorizer.filter(
                orig_image=pil_in,
                filtered_image=pil_in,
                render_factor=render_factor,
                post_process=post_process,
            )
            pil_out = _adjust_saturation(pil_out, saturation)
            frames.append(_pil_to_tensor(pil_out))
        return (torch.stack(frames, dim=0),)


class DeOldifyClearCache:
    """Clears the in-memory DeOldify model cache to free VRAM."""

    CATEGORY = "image/color"
    RETURN_TYPES = ()
    FUNCTION = "clear"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def clear(self):
        _colorizer_cache.clear()
        torch.cuda.empty_cache()
        logging.info("[DeOldify] Model cache cleared.")
        return {}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "DeOldifyColorize":   DeOldifyColorize,
    "DeOldifyClearCache": DeOldifyClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeOldifyColorize":   "DeOldify Colorize",
    "DeOldifyClearCache": "DeOldify Clear Cache",
}
