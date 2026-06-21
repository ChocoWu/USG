"""Factory helpers to build the shared OpenCLIP model (text + image towers).

Loading priority:
  1. the local snapshot checkpoint under ``checkpoints/openclip/CLIP-convnext_large_d_320``
     (offline, no network) if present;
  2. otherwise download the pretrained tag into ``checkpoints/openclip`` (cached).

The text and image encoders share a single OpenCLIP model so their embedding spaces
stay aligned (needed for open-vocabulary cosine classification).
"""

import os

import open_clip

# CLIP-ConvNeXt-L (paper §3.1 / E.2: image encoder is ConvNeXt-L).
DEFAULT_MODEL = "convnext_large_d_320"
DEFAULT_PRETRAINED = "laion2b_s29b_b131k_ft_soup"
DEFAULT_CACHE = os.path.join("checkpoints", "openclip")
# direct local snapshot (as downloaded to the path the project uses)
DEFAULT_LOCAL_CKPT = os.path.join(
    DEFAULT_CACHE, "CLIP-convnext_large_d_320", "open_clip_pytorch_model.bin"
)
IMAGE_SIZE = 320  # ConvNeXt-L input resolution (open_clip_config.json)


def build_openclip(
    model_name: str = DEFAULT_MODEL,
    pretrained: str = DEFAULT_PRETRAINED,
    cache_dir: str = DEFAULT_CACHE,
    local_checkpoint: str = None,
):
    """Create the OpenCLIP model, preferring the local checkpoint if available.

    Returns (model, preprocess).
    """
    ckpt = local_checkpoint or DEFAULT_LOCAL_CKPT
    if os.path.isfile(ckpt):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=ckpt
        )
    else:
        os.makedirs(cache_dir, exist_ok=True)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir
        )
    model.eval()
    return model, preprocess


def get_tokenizer(model_name: str = DEFAULT_MODEL):
    return open_clip.get_tokenizer(model_name)