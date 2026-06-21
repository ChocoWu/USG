"""Video frame reading for PVSG (decode specific frame indices from an mp4).

Uses PyAV. ``av_frame_reader`` matches the ``frame_reader`` signature expected by
PVSGDataset: ``(video_path, frame_idxs, preprocess) -> (T, 3, H, W)``.
"""

from typing import Callable, List, Optional

import torch
from PIL import Image


def av_frame_reader(video_path: str, frame_idxs: List[int],
                    preprocess: Optional[Callable] = None) -> torch.Tensor:
    """Decode the requested frame indices and stack them.

    Frame indices are interpreted as positions in the decoded frame sequence (the
    PVSG masks are annotated per decoded frame). Returns (T, 3, H, W); if a
    preprocess transform is given it is applied per frame (PIL -> tensor), else
    frames are returned as float tensors in [0,1], (T, 3, H, W).
    """
    import av

    wanted = sorted(set(frame_idxs))
    want_set = set(wanted)
    grabbed = {}
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        last = max(wanted)
        for i, frame in enumerate(container.decode(stream)):
            if i in want_set:
                grabbed[i] = frame.to_image()  # PIL
            if i >= last:
                break
    finally:
        container.close()

    # fall back to the nearest grabbed frame if some indices ran past the stream
    def get(i):
        if i in grabbed:
            return grabbed[i]
        return grabbed[max(grabbed)] if grabbed else Image.new("RGB", (224, 224))

    out = []
    for i in frame_idxs:
        img = get(i).convert("RGB")
        if preprocess is not None:
            out.append(preprocess(img))
        else:
            t = torch.from_numpy(__import__("numpy").asarray(img)).permute(2, 0, 1).float() / 255.0
            out.append(t)
    return torch.stack(out)