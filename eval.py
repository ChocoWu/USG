"""Evaluate USG-Par.

  conda activate usg && python eval.py --config configs/psg.yaml --ckpt <path> --task sgdet

Computes Recall@K / mean-Recall@K over the test split.
"""

import argparse

import torch
import yaml

from train import batch_inputs, build_class_embeddings, build_dataset, build_model, pick_device
from usg_par.eval.recall import SGRecallEvaluator
from usg_par.eval.setmatch import SetMatchEvaluator, build_text_pred_triplets
from usg_par.eval.triplets import build_gt_triplets, build_pred_triplets


@torch.no_grad()
def evaluate_factual(model, loader, class_emb, ds, device, score_thr=0.5):
    """Set Match over the FACTUAL test split (text SG parsing)."""
    model.eval()
    ev = SetMatchEvaluator()
    for batch in loader:
        out = model({"text": batch["tokens"].to(device)}, {"text": class_emb.to(device)})
        mo = out.per_modality["text"]
        for bi in range(mo.cls_logits.shape[0]):
            pred = build_text_pred_triplets(
                mo.cls_logits[bi].cpu(), mo.rpc_out.sub_idx[bi].cpu(),
                mo.rpc_out.obj_idx[bi].cpu(), mo.relation_logits[bi].cpu(),
                ds.object_classes, ds.predicate_classes,
                pair_scores=mo.rpc_out.scores[bi].cpu(), score_thr=score_thr)
            ev.update(pred, batch["gt_triplets"][bi])
    return ev.compute()


@torch.no_grad()
def evaluate(model, loader, class_emb, evaluator: SGRecallEvaluator, device, modality="image"):
    model.eval()
    for batch in loader:
        inputs = batch_inputs(batch, modality, device)
        out = model(inputs, {modality: class_emb.to(device)})
        mo = out.per_modality[modality]
        b = mo.cls_logits.shape[0]                 # B (image) or B*T (video, per-frame)
        for bi in range(b):
            pred = build_pred_triplets(
                mo.cls_logits[bi], mo.pred_masks[bi] if mo.pred_masks is not None else None,
                mo.rpc_out.sub_idx[bi], mo.rpc_out.obj_idx[bi], mo.relation_logits[bi],
            )
            gt = build_gt_triplets(
                batch["labels"][bi],
                batch["masks"][bi] if batch["masks"] is not None else None,
                batch["relations"][bi],
            )
            evaluator.update(pred, gt)
    return evaluator.compute()


def run_eval(cfg: dict, ckpt: str, task: str):
    from usg_par.encoders.builders import build_openclip, get_tokenizer

    device = pick_device(cfg["train"]["device"])
    clip_model, preprocess = build_openclip()
    name = cfg["dataset"]["name"]
    tokenizer = get_tokenizer()
    eval_split = "test" if name in ("psg", "factual") else "val"
    ds, collate_fn = build_dataset(cfg, preprocess, split=eval_split, tokenizer=tokenizer)
    modality = cfg["train"]["modality"]
    model = build_model(cfg, clip_model, modality, ds.num_predicates).to(device)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=device))

    class_emb = build_class_embeddings(model, tokenizer, ds.object_classes).to(device)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn)

    if name == "factual":
        metrics = evaluate_factual(model, loader, class_emb, ds, device,
                                   score_thr=cfg.get("eval", {}).get("pred_score_thr", 0.5))
        print("[factual] " + "  ".join(f"{k}={v*100:.2f}" for k, v in metrics.items()))
        return metrics

    evaluator = SGRecallEvaluator(
        k_list=(20, 50, 100), num_predicates=ds.num_predicates,
        use_mask=(task == "sgdet"))
    metrics = evaluate(model, loader, class_emb, evaluator, device, modality=modality)
    print(f"[{task}] " + "  ".join(f"{k}={v*100:.2f}" for k, v in metrics.items()))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/psg.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--task", default="sgdet", choices=["sgdet", "sgcls", "precls"])
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_eval(cfg, args.ckpt, args.task)


if __name__ == "__main__":
    main()