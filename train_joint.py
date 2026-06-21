"""Two-stage joint training for USG-Par.

  Stage 1: interleave the 4 single-modality datasets over the UNION vocab.
  Stage 2: load the stage-1 checkpoint, train on the multimodal pairs.

  conda activate usg
  PYTORCH_ENABLE_MPS_FALLBACK=1 python train_joint.py --config configs/joint.yaml --stage both

"""

import argparse
import os

import torch
import yaml

from train import (build_class_embeddings, cosine_warmup_scheduler, id_train_step,
                   iv_train_step, pick_device, sd_train_step, si_train_step, train_step)
from usg_par.datasets.interleave import InterleavedLoader, temperature_weights
from usg_par.datasets.unified import RemapWrapper, UnifiedVocab, build_unified_vocab
from usg_par.losses import HungarianMatcher, LossWeights

# field names to remap per dataset kind
SINGLE_FIELDS = {"label_fields": ("labels",), "relation_fields": ("relations",)}
PAIR_FIELDS = {
    "iv": (("image_labels", "video_labels"), ("image_relations", "video_relations")),
    "si": (("image_labels", "text_labels"), ("image_relations", "text_relations")),
    "id": (("image_labels", "point_labels"), ("image_relations", "point_relations")),
    "sd": (("text_labels", "point_labels"), ("text_relations", "point_relations")),
}
PAIR_STEP = {"iv": iv_train_step, "id": id_train_step, "si": si_train_step, "sd": sd_train_step}


# --------------------------------------------------------------------------- #
# dataset construction
# --------------------------------------------------------------------------- #
def _single_dataset(modality: str, spec: dict, preprocess, tokenizer):
    name = spec["name"]
    if name == "psg":
        from usg_par.datasets.psg import PSGDataset, psg_collate
        ds = PSGDataset(spec["ann"], image_root=spec["image_root"], preprocess=preprocess,
                        split=spec["split"], mask_size=tuple(spec["mask_size"]))
        return ds, psg_collate
    if name == "pvsg":
        from usg_par.datasets.pvsg import PVSGDataset, pvsg_collate
        from usg_par.datasets.video_io import av_frame_reader
        ds = PVSGDataset(spec["ann"], data_root=spec["data_root"], split=spec["split"],
                         num_frames=spec["num_frames"], mask_size=tuple(spec["mask_size"]),
                         preprocess=preprocess, frame_reader=av_frame_reader)
        return ds, pvsg_collate
    if name == "factual":
        from usg_par.datasets.factual import FACTUALDataset, build_factual_vocab, factual_collate
        obj, pred = build_factual_vocab(spec["train_csv"])         # vocab always from train
        csv = spec.get("test_csv", spec["train_csv"])              # rows: eval uses test_csv
        ds = FACTUALDataset(csv, tokenizer=tokenizer, object_classes=obj, predicate_classes=pred)
        return ds, factual_collate
    if name == "3ddsg":
        from usg_par.datasets.threeddsg import Scan3DSSGDataset, threeddsg_collate
        ds = Scan3DSSGDataset(spec["rscan_root"], spec["ssg_root"], spec["split"],
                              num_points=spec["num_points"])
        return ds, threeddsg_collate
    raise ValueError(name)


def _pair_dataset(key: str, spec: dict, preprocess, tokenizer):
    from usg_par.datasets.video_io import av_frame_reader
    if key == "iv":
        from usg_par.datasets.iv_pairs import IVPairDataset, iv_collate
        return IVPairDataset(spec["recipe"], "data/pvsg/pvsg.json", "data/pvsg",
                             preprocess=preprocess, frame_reader=av_frame_reader), iv_collate
    if key == "id":
        from usg_par.datasets.id_pairs import IDPairDataset, id_collate
        return IDPairDataset(spec["recipe"], "data/3DSG/3RScan", "data/3DSG/3DSSG",
                             preprocess=preprocess), id_collate
    if key == "si":
        from usg_par.datasets.si_pairs import SIPairDataset, si_collate
        return SIPairDataset(spec["recipe"], "data/psg/psg.json", "data/psg/coco",
                             preprocess=preprocess, tokenizer=tokenizer,
                             split=spec.get("split", "train")), si_collate
    if key == "sd":
        from usg_par.datasets.sd_pairs import SDPairDataset, sd_collate
        return SDPairDataset(spec["recipe"], "data/3DSG/3RScan", "data/3DSG/3DSSG",
                             tokenizer=tokenizer), sd_collate
    raise ValueError(key)


def build_unified_model(cfg, clip_model, modalities, num_predicates, device):
    from usg_par.model import USGPar
    m = cfg["model"]
    return USGPar(
        clip_model, modalities=tuple(modalities), dim=m["dim"], num_queries=m["num_queries"],
        num_predicates=num_predicates, num_scales=m["num_scales"],
        mask_decoder_layers=m["mask_decoder_layers"], rpc_layers=m["rpc_layers"],
        relation_layers=m["relation_layers"], top_k=m["top_k"],
        point_pretrained=m.get("point_pretrained"), point_in_dim=m.get("point_in_dim", 6),
    ).to(device)


def _loader(ds, collate, bs, num_workers):
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                       num_workers=num_workers, collate_fn=collate)


# --------------------------------------------------------------------------- #
# Stage 1: interleave 4 single-modality datasets over the union vocab
# --------------------------------------------------------------------------- #
def run_stage1(cfg):
    from usg_par.encoders.builders import build_openclip, get_tokenizer
    device = pick_device(cfg["train"]["device"])
    clip_model, preprocess = build_openclip()
    tokenizer = get_tokenizer()

    specs = cfg["stage1"]["datasets"]
    datasets, collates = {}, {}
    for modality, spec in specs.items():
        datasets[modality], collates[modality] = _single_dataset(modality, spec, preprocess, tokenizer)
        print(f"[stage1] {modality}: {len(datasets[modality])} items ({spec['name']})")

    uv = build_unified_vocab(
        {m: d.object_classes for m, d in datasets.items()},
        {m: d.predicate_classes for m, d in datasets.items()})
    print(f"[stage1] union vocab: {uv.num_object_classes} obj / {uv.num_predicates} pred")

    loaders = {}
    for modality, ds in datasets.items():
        wrapped = RemapWrapper(ds, uv.obj_remap[modality], uv.pred_remap[modality], **SINGLE_FIELDS)
        loaders[modality] = _loader(wrapped, collates[modality], specs[modality]["batch_size"],
                                    cfg["train"]["num_workers"])

    model = build_unified_model(cfg, clip_model, datasets.keys(), uv.num_predicates, device)
    class_emb = build_class_embeddings(model, tokenizer, uv.object_classes).to(device)
    return _run_interleaved(cfg, "stage1", model, class_emb, loaders, uv, device,
                            train_fn=lambda m, batch, **kw: train_step(*kw["args"], modality=m),
                            ckpt_path=cfg["stage1"]["ckpt"], steps=cfg["stage1"]["steps"])


# --------------------------------------------------------------------------- #
# Stage 2: load stage-1 ckpt, train on the 4 multimodal pairs
# --------------------------------------------------------------------------- #
def run_stage2(cfg, stage1_ckpt):
    from usg_par.encoders.builders import build_openclip, get_tokenizer
    device = pick_device(cfg["train"]["device"])
    clip_model, preprocess = build_openclip()
    tokenizer = get_tokenizer()

    ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    uv = UnifiedVocab(**ckpt["unified_vocab"])
    modalities = ckpt["modalities"]
    model = build_unified_model(cfg, clip_model, modalities, uv.num_predicates, device)
    model.load_state_dict(ckpt["model"])
    class_emb = build_class_embeddings(model, tokenizer, uv.object_classes).to(device)
    print(f"[stage2] loaded {stage1_ckpt} | union {uv.num_object_classes} obj / {uv.num_predicates} pred")

    loaders = {}
    for key, spec in cfg["stage2"]["pairs"].items():
        ds, collate = _pair_dataset(key, spec, preprocess, tokenizer)
        base = spec["base"]                                  # which stage-1 vocab to remap with
        lf, rf = PAIR_FIELDS[key]
        wrapped = RemapWrapper(ds, uv.obj_remap[base], uv.pred_remap[base],
                               label_fields=lf, relation_fields=rf)
        loaders[key] = _loader(wrapped, collate, spec["batch_size"], cfg["train"]["num_workers"])
        print(f"[stage2] {key}: {len(ds)} pairs (base={base})")

    return _run_interleaved(cfg, "stage2", model, class_emb, loaders, uv, device,
                            train_fn=lambda k, batch, **kw: PAIR_STEP[k](*kw["args"]),
                            ckpt_path=cfg["stage2"]["ckpt"], steps=cfg["stage2"]["steps"])


def _run_interleaved(cfg, tag, model, class_emb, loaders, uv, device, train_fn, ckpt_path, steps):
    weights = temperature_weights({k: len(l.dataset) for k, l in loaders.items()},
                                  cfg["train"]["temperature"])
    il = InterleavedLoader(loaders, weights, steps)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = cosine_warmup_scheduler(opt, cfg["train"]["warmup_steps"], steps)
    matcher, lw = HungarianMatcher(), LossWeights()
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    print(f"[{tag}] interleave weights: " + " ".join(f"{k}={w:.2f}" for k, w in weights.items()))
    for step, (key, batch) in enumerate(il):
        args = (model, batch, class_emb, uv.num_predicates, matcher, lw, opt, sched, device,
                cfg["train"]["grad_clip"])
        parts = train_fn(key, batch, args=args)
        if step % cfg["train"]["log_every"] == 0:
            print(f"[{tag}] step {step} ({key}) | " +
                  " ".join(f"{k}={v:.4f}" for k, v in parts.items()))
    torch.save({"model": model.state_dict(),
                "modalities": ["image", "video", "text", "point"],   # unified model always has all 4
                "unified_vocab": uv.__dict__}, ckpt_path)
    print(f"[{tag}] saved {ckpt_path}")
    return ckpt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/joint.yaml")
    ap.add_argument("--stage", default="both", choices=["1", "2", "both"])
    ap.add_argument("--stage1-ckpt", default=None, help="for --stage 2: stage-1 checkpoint")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    ckpt = args.stage1_ckpt
    if args.stage in ("1", "both"):
        ckpt = run_stage1(cfg)
    if args.stage in ("2", "both"):
        run_stage2(cfg, ckpt or cfg["stage1"]["ckpt"])


if __name__ == "__main__":
    main()