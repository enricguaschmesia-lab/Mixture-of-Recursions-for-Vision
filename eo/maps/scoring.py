"""GEO-Bench-2's segmentation metric, boundary accuracy and code statistics (A2 Step 1).

The metric is torchmetrics' MulticlassJaccardIndex(num_classes, ignore_index, average="macro"), the
metric GEO-Bench-2 reads (terratorch's segmentation task), accumulated over a whole split:
  - pixels whose target is the ignored class are dropped; predictions are never the ignored class;
  - per class, IoU = tp / (tp + fp + fn);
  - the macro mean weighs every class 1, except the ignored class and classes with neither target
    nor prediction pixels (weight 0). A class predicted but absent from the targets scores 0 and
    counts. (Step 0's scorer averaged over target classes only; on whole folds the two agree, since
    every class is present.)
`Confusion` accumulates a confusion matrix on the GPU, so many decoding rules can be read in one pass.
Z3 holds `miou` equal to torchmetrics on synthetic maps with void; Z2 holds the two paths equal on a
real ceiling.

Boundary pixels are Step 0's: scored pixels with a 4-neighbour of another label (`rgb_map_ceiling.boundary`).
"""
from __future__ import annotations

import numpy as np
import torch


def boundary(labels: torch.Tensor) -> torch.Tensor:
    """(N, H, W) -> bool, a pixel whose 4-neighbour differs (any label, void included). Step 0's rule."""
    e = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
    d = labels[..., 1:, :] != labels[..., :-1, :]
    e[..., 1:, :] |= d
    e[..., :-1, :] |= d
    d = labels[..., :, 1:] != labels[..., :, :-1]
    e[..., :, 1:] |= d
    e[..., :, :-1] |= d
    return e


class Confusion:
    """Running confusion matrix (target x prediction) and boundary / interior counts."""

    def __init__(self, n_classes: int, ignore: int | None, device="cpu"):
        self.n, self.ignore = n_classes, ignore
        self.cm = torch.zeros(n_classes * n_classes, dtype=torch.int64, device=device)
        self.b_ok = self.b_n = self.i_ok = self.i_n = 0

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        gt, pred = gt.long(), pred.long().to(gt.device)
        if self.ignore is not None and bool((pred == self.ignore).any()):
            raise ValueError("a prediction is the ignored class, which is never predicted")
        valid = gt != self.ignore if self.ignore is not None else torch.ones_like(gt, dtype=torch.bool)
        self.cm += torch.bincount((gt[valid] * self.n + pred[valid]).to(self.cm.device),
                                  minlength=self.n * self.n)
        b = boundary(gt)
        ok = pred == gt
        self.b_ok += int((ok & valid & b).sum())
        self.b_n += int((valid & b).sum())
        self.i_ok += int((ok & valid & ~b).sum())
        self.i_n += int((valid & ~b).sum())

    def matrix(self) -> np.ndarray:
        return self.cm.view(self.n, self.n).cpu().numpy()

    def result(self) -> dict:
        cm = self.matrix().astype(np.float64)
        tp = np.diag(cm)
        denom = cm.sum(0) + cm.sum(1) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(denom > 0, tp / np.where(denom > 0, denom, 1), 0.0)
        w = np.ones(self.n)
        if self.ignore is not None:
            w[self.ignore] = 0
        w[(cm.sum(0) + cm.sum(1)) == 0] = 0
        present = cm.sum(1) > 0
        if self.ignore is not None:
            present[self.ignore] = False
        return {
            "mIoU": float((w * iou).sum() / w.sum()),
            "mIoU_target_classes": float(iou[present].mean()),     # Step 0's averaging
            "pixel_acc": float(tp.sum() / cm.sum()),
            "boundary_acc": self.b_ok / max(self.b_n, 1),
            "interior_acc": self.i_ok / max(self.i_n, 1),
            "boundary_share": self.b_n / max(self.b_n + self.i_n, 1),
            "iou_per_class": [float(v) if w[i] > 0 else None for i, v in enumerate(iou)],
            "classes_scored": int(w.sum()),
            "pixels": int(cm.sum()),
        }


def miou(gt, pred, n_classes: int, ignore: int | None) -> float:
    """One-shot mIoU of two label arrays (numpy or torch)."""
    c = Confusion(n_classes, ignore)
    c.update(torch.as_tensor(np.asarray(gt)), torch.as_tensor(np.asarray(pred)))
    return c.result()["mIoU"]


def torchmetrics_miou(gt, pred, n_classes: int, ignore: int | None) -> float:
    """The same through torchmetrics (the second scoring path of Z2 and the reference of Z3)."""
    from torchmetrics.classification import MulticlassJaccardIndex
    m = MulticlassJaccardIndex(num_classes=n_classes, ignore_index=ignore, average="macro")
    m.update(torch.as_tensor(np.asarray(pred)).long(), torch.as_tensor(np.asarray(gt)).long())
    return float(m.compute())


def code_stats(read_idx: np.ndarray, ref_idx: np.ndarray | None, n_codes: int) -> dict:
    """Distinct codes and entropy of the read split's tokens, and the share of its tokens whose code
    never occurs in the reference (training) tokens. Step 0's `code_stats`, with the entropy added."""
    t = np.asarray(read_idx).ravel().astype(np.int64)
    ut, ct = np.unique(t, return_counts=True)
    p = ct / ct.sum()
    out = {
        "n_codes": int(n_codes),
        "read_tokens": int(t.size),
        "read_distinct": int(len(ut)),
        "read_entropy_bits": float(-(p * np.log2(p)).sum()),
        "top1000_code_mass": float(np.sort(ct)[::-1][:1000].sum() / t.size),
    }
    if ref_idx is not None:
        r = np.asarray(ref_idx).ravel().astype(np.int64)
        ur, cr = np.unique(r, return_counts=True)
        pr = cr / cr.sum()
        seen = np.isin(t, ur)
        out.update({
            "ref_tokens": int(r.size),
            "ref_distinct": int(len(ur)),
            "ref_entropy_bits": float(-(pr * np.log2(pr)).sum()),
            "unseen_token_share": float(1 - seen.mean()),
            "unseen_code_share": float(1 - np.isin(ut, ur).mean()),
        })
    return out
