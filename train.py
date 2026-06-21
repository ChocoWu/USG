"""Train USG-Par on a single-modality SGDet dataset (PSG by default).

  conda activate usg && python train.py --config configs/psg.yaml

"""

import argparse
import math
import os

import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR

from usg_par.datasets.psg import PSGDataset, psg_collate
from usg_par.datasets.pvsg import PVSGDataset, pvsg_collate
from usg_par.losses import HungarianMatcher, LossWeights
from usg_par.training.loss_assembly import compute_single_modality_losses


def build_dataset(cfg: dict, preprocess, split: str, tokenizer=None):
    """Build a PSG / PVSG / FACTUAL dataset (+ its collate fn) from config."""
    name = cfg["dataset"]["name"]
    if name == "3ddsg":
        from usg_par.datasets.threeddsg import Scan3DSSGDataset, threeddsg_collate
        split_file = cfg["dataset"]["split"][{"train": "train", "val": "val", "test": "test"}[split]]
        ds = Scan3DSSGDataset(cfg["dataset"]["rscan_root"], cfg["dataset"]["ssg_root"],
                              split_file, num_points=cfg["dataset"]["num_points"])
        return ds, threeddsg_collate
    if name == "factual":
        from usg_par.datasets.factual import FACTUALDataset, build_factual_vocab, factual_collate
        obj, pred = build_factual_vocab(cfg["dataset"]["train_csv"])   # shared vocab from train
        csv_key = {"train": "train_csv", "val": "dev_csv", "dev": "dev_csv", "test": "test_csv"}[split]
        ds = FACTUALDataset(cfg["dataset"][csv_key], tokenizer=tokenizer,
                            object_classes=obj, predicate_classes=pred)
        return ds, factual_collate
    if name == "pvsg":
        from usg_par.datasets.video_io import av_frame_reader
        ds = PVSGDataset(
            cfg["dataset"]["ann"], data_root=cfg["dataset"]["data_root"], split=split,
            num_frames=cfg["dataset"]["num_frames"], mask_size=tuple(cfg["dataset"]["mask_size"]),
            preprocess=preprocess, frame_reader=av_frame_reader,
        )
        return ds, pvsg_collate
    ds = PSGDataset(
        cfg["dataset"]["ann"], image_root=cfg["dataset"]["image_root"], preprocess=preprocess,
        split=split, mask_size=tuple(cfg["dataset"]["mask_size"]),
    )
    return ds, psg_collate


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to 0."""
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return LambdaLR(optimizer, fn)


def build_model(cfg: dict, clip_model, modality: str, num_predicates: int):
    """Construct USGPar for the configured modality (incl. point-encoder pretrained)."""
    from usg_par.model import USGPar
    m = cfg["model"]
    point_kwargs = {}
    if modality == "point":
        point_kwargs = dict(
            point_pretrained=m.get("point_pretrained"),
            point_in_dim=m.get("point_in_dim", 6),
            point_freeze_encoder=m.get("point_freeze_encoder", False),
        )
    return USGPar(
        clip_model, modalities=(modality,), dim=m["dim"], num_queries=m["num_queries"],
        num_predicates=num_predicates, num_scales=m["num_scales"],
        mask_decoder_layers=m["mask_decoder_layers"], rpc_layers=m["rpc_layers"],
        relation_layers=m["relation_layers"], top_k=m["top_k"], **point_kwargs,
    )


def build_class_embeddings(model, tokenizer, object_classes):
    """Frozen open-vocab class-name embeddings (133, d)."""
    device = next(model.parameters()).device
    ids = tokenizer(object_classes).to(device)
    return model.text_encoder.encode_class_names(ids)


def to_image_batch(batch):
    """Normalize a PSG (images) or PVSG (frames B,T,3,H,W) batch to an image batch.

    For video the T frames are folded into the batch (B*T), matching the per-frame GT
    order produced by pvsg_collate (row-major video,frame). Per-frame VSG reuses the
    image pipeline; the temporal encoder (F_temp) is a later refinement.
    """
    if batch.get("frames") is not None:
        f = batch["frames"]
        b, t = f.shape[:2]
        return {"images": f.reshape(b * t, *f.shape[2:]),
                "labels": batch["labels"], "masks": batch["masks"],
                "relations": batch["relations"]}
    return batch


def batch_inputs(batch, modality, device):
    """Build the model input dict for one batch.

    Video is fed natively as (B,T,3,H,W) so the model can apply the temporal encoder
    F_temp; the per-frame GT (B*T) is produced by pvsg_collate in matching order.
    """
    if modality == "video":
        return {"video": batch["frames"].to(device)}
    if modality == "text":
        return {"text": batch["tokens"].to(device)}
    if modality == "point":
        return {"point": batch["points"].to(device)}
    return {"image": batch["images"].to(device)}


def train_step(model, batch, class_emb, num_predicates, matcher, loss_weights,
               optimizer, scheduler, device, grad_clip, modality="image"):
    """Run one optimization step. Returns the loss breakdown (floats)."""
    model.train()
    inputs = batch_inputs(batch, modality, device)
    gt_labels = [t.to(device) for t in batch["labels"]]
    gt_masks = [t.to(device) for t in batch["masks"]] if batch["masks"] is not None else None
    gt_rel = [t.to(device) for t in batch["relations"]]

    out = model(inputs, {modality: class_emb.to(device)})
    total, parts = compute_single_modality_losses(
        out, modality, gt_labels, gt_masks, gt_rel, num_predicates,
        matcher=matcher, loss_weights=loss_weights,
    )
    optimizer.zero_grad()
    total.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {k: float(v.detach()) for k, v in parts.items()}


def iv_train_step(model, batch, class_emb, num_predicates, matcher, loss_weights,
                  optimizer, scheduler, device, grad_clip):
    """One I-V multimodal step. The image is repeated to B*T so it aligns with each
    video frame (per-frame cross-modal association). Returns the loss breakdown."""
    from usg_par.datasets.iv_pairs import build_iv_association
    from usg_par.training.loss_assembly import compute_multimodal_losses

    model.train()
    b, t = batch["images"].shape[0], batch["num_frames"]
    frames = batch["frames"].to(device)

    def rep(field):  # repeat each video's image-GT T times -> B*T, matching the folded video order
        return [batch[field][i].to(device) for i in range(b) for _ in range(t)]

    gt = {
        "image": {"labels": rep("image_labels"), "masks": rep("image_masks"),
                  "relations": rep("image_relations")},
        "video": {"labels": [x.to(device) for x in batch["video_labels"]],
                  "masks": [x.to(device) for x in batch["video_masks"]],
                  "relations": [x.to(device) for x in batch["video_relations"]]},
    }
    obj_assoc = [build_iv_association(batch["image_object_ids"][i], batch["video_object_ids"][i * t + j])
                 for i in range(b) for j in range(t)]

    # encode image once (B) -> repeat features to B*T; video folded to B*T
    feats = model.encode_iv(batch["images"].to(device), frames)
    out = model.core(feats, {"image": class_emb.to(device), "video": class_emb.to(device)})
    total, parts = compute_multimodal_losses(
        out, gt, obj_assoc, pair=("image", "video"),
        num_predicates_per_modality={"image": num_predicates, "video": num_predicates},
        matcher=matcher, loss_weights=loss_weights)

    optimizer.zero_grad()
    total.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {k: float(v.detach()) for k, v in parts.items()}


def id_train_step(model, batch, class_emb, num_predicates, matcher, loss_weights,
                  optimizer, scheduler, device, grad_clip):
    """One I-D multimodal step. Both modalities are batch B (image + point, no folding),
    so the associator works directly. Returns the loss breakdown."""
    from usg_par.datasets.id_pairs import build_iv_association
    from usg_par.training.loss_assembly import compute_multimodal_losses

    model.train()
    b = batch["images"].shape[0]
    to = lambda field: [x.to(device) for x in batch[field]]
    gt = {
        "image": {"labels": to("image_labels"), "masks": to("image_masks"),
                  "relations": to("image_relations")},
        "point": {"labels": to("point_labels"), "masks": to("point_masks"),
                  "relations": to("point_relations")},
    }
    obj_assoc = [build_iv_association(batch["image_object_ids"][i], batch["point_object_ids"][i])
                 for i in range(b)]

    out = model({"image": batch["images"].to(device), "point": batch["points"].to(device)},
                {"image": class_emb.to(device), "point": class_emb.to(device)})
    total, parts = compute_multimodal_losses(
        out, gt, obj_assoc, pair=("image", "point"),
        num_predicates_per_modality={"image": num_predicates, "point": num_predicates},
        matcher=matcher, loss_weights=loss_weights)

    optimizer.zero_grad()
    total.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {k: float(v.detach()) for k, v in parts.items()}


def si_train_step(model, batch, class_emb, num_predicates, matcher, loss_weights,
                  optimizer, scheduler, device, grad_clip):
    """One S-I multimodal step (image + text, batch B). Text has no masks (mask-free
    path); association is category-level; includes L_cons. Returns the breakdown."""
    from usg_par.datasets.si_pairs import build_category_association
    from usg_par.training.loss_assembly import compute_multimodal_losses

    model.train()
    b = batch["images"].shape[0]
    to = lambda field: [x.to(device) for x in batch[field]]
    img_masks = to("image_masks") if batch["image_masks"] is not None else None
    gt = {
        "image": {"labels": to("image_labels"), "masks": img_masks,
                  "relations": to("image_relations")},
        "text": {"labels": to("text_labels"), "masks": None,
                 "relations": to("text_relations")},
    }
    # object association in pair order (image, text): (M_image, M_text)
    obj_assoc = [build_category_association(batch["image_labels"][i], batch["text_labels"][i])
                 for i in range(b)]

    out = model({"image": batch["images"].to(device), "text": batch["tokens"].to(device)},
                {"image": class_emb.to(device), "text": class_emb.to(device)})
    total, parts = compute_multimodal_losses(
        out, gt, obj_assoc, pair=("image", "text"),
        num_predicates_per_modality={"image": num_predicates, "text": num_predicates},
        matcher=matcher, loss_weights=loss_weights)

    optimizer.zero_grad()
    total.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {k: float(v.detach()) for k, v in parts.items()}


def sd_train_step(model, batch, class_emb, num_predicates, matcher, loss_weights,
                  optimizer, scheduler, device, grad_clip):
    """One S-D multimodal step (point + text, batch B). Text has no masks; point has
    3D masks; category-level association; includes L_cons. Returns the breakdown."""
    from usg_par.datasets.si_pairs import build_category_association
    from usg_par.training.loss_assembly import compute_multimodal_losses

    model.train()
    b = batch["points"].shape[0]
    to = lambda field: [x.to(device) for x in batch[field]]
    gt = {
        "point": {"labels": to("point_labels"), "masks": to("point_masks"),
                  "relations": to("point_relations")},
        "text": {"labels": to("text_labels"), "masks": None,
                 "relations": to("text_relations")},
    }
    obj_assoc = [build_category_association(batch["point_labels"][i], batch["text_labels"][i])
                 for i in range(b)]

    out = model({"point": batch["points"].to(device), "text": batch["tokens"].to(device)},
                {"point": class_emb.to(device), "text": class_emb.to(device)})
    total, parts = compute_multimodal_losses(
        out, gt, obj_assoc, pair=("point", "text"),
        num_predicates_per_modality={"point": num_predicates, "text": num_predicates},
        matcher=matcher, loss_weights=loss_weights)

    optimizer.zero_grad()
    total.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {k: float(v.detach()) for k, v in parts.items()}


def run_training(cfg: dict):
    from usg_par.encoders.builders import build_openclip, get_tokenizer

    device = pick_device(cfg["train"]["device"])
    clip_model, preprocess = build_openclip()
    tokenizer = get_tokenizer()

    ds, collate_fn = build_dataset(cfg, preprocess, split="train", tokenizer=tokenizer)
    print(f"[{cfg['dataset']['name']}] train items: {len(ds)}")

    modality = cfg["train"]["modality"]               # image/video/text/point
    model = build_model(cfg, clip_model, modality, ds.num_predicates).to(device)

    class_emb = build_class_embeddings(model, tokenizer, ds.object_classes).to(device)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    total_steps = cfg["train"]["epochs"] * max(1, len(loader))
    scheduler = cosine_warmup_scheduler(optimizer, cfg["train"]["warmup_steps"], total_steps)
    matcher = HungarianMatcher()
    lw = LossWeights(alpha=cfg["loss"]["alpha"], gamma=cfg["loss"]["gamma"])

    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        for batch in loader:
            parts = train_step(
                model, batch, class_emb, ds.num_predicates, matcher, lw,
                optimizer, scheduler, device, cfg["train"]["grad_clip"], modality=modality,
            )
            if step % cfg["train"]["log_every"] == 0:
                print(f"epoch {epoch} step {step} | " +
                      " ".join(f"{k}={v:.4f}" for k, v in parts.items()))
            step += 1
        torch.save(model.state_dict(), os.path.join(cfg["train"]["ckpt_dir"], f"epoch{epoch}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/psg.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_training(cfg)


if __name__ == "__main__":
    main()