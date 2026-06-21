"""Evaluate the joint (two-stage) USG-Par model on every test split.

  conda activate usg
  PYTORCH_ENABLE_MPS_FALLBACK=1 python eval_joint.py --config configs/joint.yaml --ckpt checkpoints/joint/stage2.pt

Reports, over the union vocab:
  * R@K / mR@K   for image (PSG) / video (PVSG) / point (3DDSG)
  * Set Match    for text (FACTUAL)
  * Assoc Acc@5  for the four cross-modal pairs (I-V / I-D / S-I / S-D)

"""

import argparse

import torch
import yaml

from train import batch_inputs, build_class_embeddings, pick_device
from train_joint import PAIR_FIELDS, _pair_dataset, _single_dataset, build_unified_model
from usg_par.datasets.iv_pairs import build_iv_association as iv_assoc
from usg_par.datasets.si_pairs import build_category_association
from usg_par.datasets.unified import RemapWrapper, UnifiedVocab
from usg_par.eval.assoc import AssocAccEvaluator
from usg_par.eval.recall import SGRecallEvaluator
from usg_par.eval.setmatch import SetMatchEvaluator, build_text_pred_triplets
from usg_par.eval.triplets import build_gt_triplets, build_pred_triplets
from usg_par.losses import HungarianMatcher
from usg_par.training.targets import build_association_targets


def _subset(ds, limit):
    n = min(limit, len(ds)) if limit else len(ds)
    return torch.utils.data.Subset(ds, list(range(n)))


def _loader(ds, collate, bs=2):
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate)


# --------------------------------------------------------------------------- #
# single-modality
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_recall(model, loader, class_emb, modality, num_predicates, device):
    ev = SGRecallEvaluator(k_list=(20, 50, 100), num_predicates=num_predicates, use_mask=True)
    model.eval()
    for batch in loader:
        out = model(batch_inputs(batch, modality, device), {modality: class_emb})
        mo = out.per_modality[modality]
        for bi in range(mo.cls_logits.shape[0]):
            pred = build_pred_triplets(mo.cls_logits[bi].cpu(),
                                       mo.pred_masks[bi].cpu() if mo.pred_masks is not None else None,
                                       mo.rpc_out.sub_idx[bi].cpu(), mo.rpc_out.obj_idx[bi].cpu(),
                                       mo.relation_logits[bi].cpu())
            gt = build_gt_triplets(batch["labels"][bi],
                                   batch["masks"][bi] if batch["masks"] is not None else None,
                                   batch["relations"][bi])
            ev.update(pred, gt)
    return ev.compute()


@torch.no_grad()
def eval_setmatch(model, loader, class_emb, obj_names, pred_names, device, thr=0.5):
    ev = SetMatchEvaluator()
    model.eval()
    for batch in loader:
        out = model({"text": batch["tokens"].to(device)}, {"text": class_emb})
        mo = out.per_modality["text"]
        for bi in range(mo.cls_logits.shape[0]):
            pred = build_text_pred_triplets(
                mo.cls_logits[bi].cpu(), mo.rpc_out.sub_idx[bi].cpu(), mo.rpc_out.obj_idx[bi].cpu(),
                mo.relation_logits[bi].cpu(), obj_names, pred_names,
                pair_scores=mo.rpc_out.scores[bi].cpu(), score_thr=thr)
            ev.update(pred, batch["gt_triplets"][bi])
    return ev.compute()


# --------------------------------------------------------------------------- #
# cross-modal Association Accuracy
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_assoc(model, loader, class_emb, key, device, matcher):
    """Assoc Acc@5 for one pair. Handles I-V (per-frame) and the batch-B pairs."""
    ev = AssocAccEvaluator(k_list=(5,))
    model.eval()
    for batch in loader:
        if key == "iv":
            t = batch["num_frames"]
            feats = model.encode_iv(batch["images"].to(device), batch["frames"].to(device))
            out = model.core(feats, {"image": class_emb, "video": class_emb})
            a_pred = out.associations["image|video"]; n = a_pred.shape[1]
            mo_s, mo_t = out.per_modality["image"], out.per_modality["video"]   # src=image (sorted)
            for k in range(a_pred.shape[0]):
                i = k // t
                sm = matcher.match_one(mo_s.cls_logits[k].cpu(), mo_s.pred_masks[k].cpu(),
                                       batch["image_labels"][i], batch["image_masks"][i])
                tm = matcher.match_one(mo_t.cls_logits[k].cpu(), mo_t.pred_masks[k].cpu(),
                                       batch["video_labels"][k], batch["video_masks"][k])
                oa = iv_assoc(batch["image_object_ids"][i], batch["video_object_ids"][k])
                ev.update(a_pred[k].cpu(), build_association_targets(sm, tm, oa, n))
            continue

        # batch-B pairs: build inputs + per-sample obj association
        if key == "id":
            inp = {"image": batch["images"].to(device), "point": batch["points"].to(device)}
            ma, mb, lab_a, msk_a, lab_b, msk_b = "image", "point", "image_labels", "image_masks", "point_labels", "point_masks"
            obj_assoc = [iv_assoc(batch["image_object_ids"][b], batch["point_object_ids"][b])
                         for b in range(batch["images"].shape[0])]
        elif key == "si":
            inp = {"image": batch["images"].to(device), "text": batch["tokens"].to(device)}
            ma, mb, lab_a, msk_a, lab_b, msk_b = "image", "text", "image_labels", "image_masks", "text_labels", None
            obj_assoc = [build_category_association(batch["image_labels"][b], batch["text_labels"][b])
                         for b in range(batch["images"].shape[0])]
        elif key == "sd":
            inp = {"point": batch["points"].to(device), "text": batch["tokens"].to(device)}
            ma, mb, lab_a, msk_a, lab_b, msk_b = "point", "text", "point_labels", "point_masks", "text_labels", None
            obj_assoc = [build_category_association(batch["point_labels"][b], batch["text_labels"][b])
                         for b in range(batch["points"].shape[0])]
        else:
            raise ValueError(key)

        out = model(inp, {ma: class_emb, mb: class_emb})
        src, tgt = sorted((ma, mb))
        a_pred = out.associations[f"{src}|{tgt}"]; n = a_pred.shape[1]
        mo_s, mo_t = out.per_modality[src], out.per_modality[tgt]
        bcount = a_pred.shape[0]
        for k in range(bcount):
            ga = matcher.match_one(out.per_modality[ma].cls_logits[k].cpu(),
                                   _m(out.per_modality[ma].pred_masks, k), batch[lab_a][k], _g(batch, msk_a, k))
            gb = matcher.match_one(out.per_modality[mb].cls_logits[k].cpu(),
                                   _m(out.per_modality[mb].pred_masks, k), batch[lab_b][k], _g(batch, msk_b, k))
            oa = obj_assoc[k]                                  # (M_a, M_b)
            oa = oa if src == ma else oa.t()                  # orient to (src, tgt)
            sm, tm = (ga, gb) if src == ma else (gb, ga)
            ev.update(a_pred[k].cpu(), build_association_targets(sm, tm, oa, n))
    return ev.compute()


def _m(masks, k):
    return masks[k].cpu() if masks is not None else None


def _g(batch, field, k):
    return batch[field][k] if field is not None and batch.get(field) is not None else None


# --------------------------------------------------------------------------- #
def run_eval_joint(cfg, ckpt_path):
    from usg_par.encoders.builders import build_openclip, get_tokenizer
    device = pick_device(cfg["train"]["device"])
    clip_model, preprocess = build_openclip()
    tokenizer = get_tokenizer()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    uv = UnifiedVocab(**ckpt["unified_vocab"])
    model = build_unified_model(cfg, clip_model, ckpt["modalities"], uv.num_predicates, device)
    model.load_state_dict(ckpt["model"])
    class_emb = build_class_embeddings(model, tokenizer, uv.object_classes).to(device)
    matcher = HungarianMatcher()
    limit = cfg["eval"].get("limit")
    print(f"[eval] {ckpt_path} | union {uv.num_object_classes} obj / {uv.num_predicates} pred")
    results = {}

    # single-modality
    for modality, spec in cfg["eval"]["single"].items():
        ds, collate = _single_dataset(modality, spec, preprocess, tokenizer)
        wrapped = RemapWrapper(ds, uv.obj_remap[modality], uv.pred_remap[modality])
        loader = _loader(_subset(wrapped, limit), collate)
        if spec["metric"] == "setmatch":
            m = eval_setmatch(model, loader, class_emb, uv.object_classes, uv.predicate_classes, device)
        else:
            m = eval_recall(model, loader, class_emb, modality, uv.num_predicates, device)
        results[modality] = m
        print(f"[eval] {modality:6s} ({spec['name']}): " + "  ".join(f"{k}={v*100:.2f}" for k, v in m.items()))

    # cross-modal assoc
    for key, spec in cfg["eval"]["pairs"].items():
        ds, collate = _pair_dataset(key, spec, preprocess, tokenizer)
        lf, rf = PAIR_FIELDS[key]
        wrapped = RemapWrapper(ds, uv.obj_remap[spec["base"]], uv.pred_remap[spec["base"]],
                               label_fields=lf, relation_fields=rf)
        loader = _loader(_subset(wrapped, limit), collate, bs=1)
        m = eval_assoc(model, loader, class_emb, key, device, matcher)
        results[key] = m
        print(f"[eval] {key:6s} (assoc): " + "  ".join(f"{k}={v*100:.2f}" for k, v in m.items()))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/joint.yaml")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    run_eval_joint(yaml.safe_load(open(args.config)), args.ckpt)


if __name__ == "__main__":
    main()