"""A2 Step 4.2's Route C: a U-Net head on the decoder's hidden states (docs/plans/a2_segmentation.md, Step 4.2, item 3).

The protocol-matched reference: TerraMind's released fine-tuning head (terramind_v1_base_*.yaml: necks SelectIndices,
ReshapeTokensToImage, LearnedInterpolateToPyramidal; decoder UNetDecoder [512, 256, 128, 64]; head_dropout 0.1), on the
hidden states of our decoder instead of a ViT's.

  input     [BO_S2L2A s1..s196 EO][BO_S2L2A s1..s196 EO]: the view's S2 tokens twice, 396 tokens at positions 0..395
            (eo/data/seg_head_dataset.py). The second copy sits at the base's 198-token lattice offset, and each of its
            tokens sees the whole first copy under the causal mask.
  features  the hidden states of the second copy's 196 image tokens (sequence positions 199..394) at HF's
            hidden_states[7, 14, 21, 29]: the outputs of layers 7, 14 and 21 and of layer 29 after the final RMSNorm, the
            quarters of 29 layers as TerraMind takes [2, 5, 8, 11] of 12. Each is reshaped to (576, 14, 14), the
            tokenizer's raster order (terratorch's ReshapeTokensToImage). Measured before the design was written: no
            massive activation at these positions (the attention sink is position 0), so no extra normalization.
  neck      terratorch 1.2.11's LearnedInterpolateToPyramidal: (144, 56, 56), (288, 28, 28), (576, 14, 14), (576, 7, 7)
  decoder   terratorch's UNetDecoder [512, 256, 128, 64] on segmentation_models_pytorch 0.5.0's UnetDecoder: each block
            upsamples (nearest) to the next skip's size, concatenates it, then two 3x3 conv + BN + ReLU; the first input
            is duplicated as terratorch does, so the last block ends at 56 x 56; initialize_decoder's init
  head      terratorch's SegmentationHead: dropout 0.1, a 1x1 conv to the task's classes; then the bilinear resize to
            the 224 canvas (PixelWiseModel's rescale)
  loss      pixel CE at native resolution: the 224 logits pooled to the native grid by eo/maps/decode.to_native (the
            render's adjoint; PASTIS 224 -> 128, CloudSen12's 224-px crops are native), void ignored on PASTIS.

What trains: the recipe's LoRA adapters (eo/finetune/lora.py) at the run's learning_rate, and the head at seg_head.lr
(SegHeadTrainer's second parameter group). The map rows are frozen: Route C writes no map token.

THE BATCH: input_ids, attention_mask, position_ids, modality_ids and seg_labels. `seg_labels` (the view's native label
map) takes the place of `labels`: the wrapper's forward names it, so it is not the stray sixth key CLAUDE.md warns of,
and with no `labels` key the trainer's per-modality CE has nothing to read. `accepts_loss_kwargs = False`, so HF divides
each micro-batch's mean CE by the accumulation steps (it would not if the forward took **kwargs).

Wired into pretrain.py by `seg_head.enable` (wrap_model after the fine-tune init, SegHeadTrainer for the trainer).
Gate: eo/scripts/verify_seg_routes.py (Z21-Z23). Runs in `.venv` (terratorch is not installable there, hence the copy;
Z21 holds it equal to terratorch's own modules in `a2seg`).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

LAYERS = (7, 14, 21, 29)
CHANNELS = (512, 256, 128, 64)
COPY_START = 199                  # the second copy's first image token: [BO s1..s196 EO] is 0..197, its BO is 198
GRID = 14
CANVAS = 224


# ------------------------------------------------------------------------------------------ terratorch / smp, copied
class LearnedInterpolateToPyramidal(nn.Module):
    """terratorch.models.necks.LearnedInterpolateToPyramidal (1.2.11)."""

    def __init__(self, channel_list: Sequence[int]):
        super().__init__()
        if len(channel_list) != 4:
            raise ValueError("This class can only handle exactly 4 input embeddings")
        c = list(channel_list)
        self.fpn1 = nn.Sequential(nn.ConvTranspose2d(c[0], c[0] // 2, 2, 2), nn.BatchNorm2d(c[0] // 2), nn.GELU(),
                                  nn.ConvTranspose2d(c[0] // 2, c[0] // 4, 2, 2))
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(c[1], c[1] // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.embedding_dim = [c[0] // 4, c[1] // 2, c[2], c[3]]

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.fpn1(features[0]), self.fpn2(features[1]), self.fpn3(features[2]), self.fpn4(features[3])]


class Conv2dReLU(nn.Sequential):
    """segmentation_models_pytorch.base.modules.Conv2dReLU with use_norm="batchnorm" (0.5.0): conv (no bias), BN, ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class UnetDecoderBlock(nn.Module):
    """smp 0.5.0's UnetDecoderBlock (no attention, nearest interpolation)."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, 3, 1)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, 1)

    def forward(self, x: torch.Tensor, h: int, w: int, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, size=(h, w), mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UnetDecoder(nn.Module):
    """smp 0.5.0's UnetDecoder (add_center_block False, so the center is the identity)."""

    def __init__(self, encoder_channels: Sequence[int], decoder_channels: Sequence[int]):
        super().__init__()
        enc = list(encoder_channels)[1:][::-1]
        head = enc[0]
        ins = [head] + list(decoder_channels[:-1])
        skips = list(enc[1:]) + [0]
        self.blocks = nn.ModuleList([UnetDecoderBlock(i, s, o) for i, s, o in zip(ins, skips, decoder_channels)])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        shapes = [f.shape[2:] for f in features][::-1]
        feats = features[1:][::-1]
        x, skips = feats[0], feats[1:]
        for i, block in enumerate(self.blocks):
            h, w = shapes[i + 1]
            x = block(x, int(h), int(w), skips[i] if i < len(skips) else None)
        return x


def initialize_decoder(module: nn.Module) -> None:
    """segmentation_models_pytorch.base.initialization.initialize_decoder (0.5.0)."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class UNetDecoder(nn.Module):
    """terratorch.models.decoders.unet_decoder.UNetDecoder (1.2.11): smp's decoder with the first input duplicated."""

    def __init__(self, embed_dim: Sequence[int], channels: Sequence[int]):
        super().__init__()
        if len(embed_dim) != len(channels):
            raise ValueError("channels should have the same length as embed_dim")
        self.decoder = UnetDecoder([embed_dim[0], *embed_dim], channels)
        initialize_decoder(self.decoder)
        self.out_channels = channels[-1]

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.decoder([x[0].clone(), *x])


class SegmentationHead(nn.Module):
    """terratorch.models.heads.segmentation_head.SegmentationHead with channel_list None: dropout, then a 1x1 conv."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.head = nn.Sequential(nn.Identity(), nn.Identity() if dropout == 0 else nn.Dropout(dropout),
                                  nn.Conv2d(in_channels, num_classes, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class UNetHead(nn.Module):
    """Four (B, 196, D) token maps -> (B, n_classes, canvas, canvas) logits."""

    def __init__(self, dim: int, n_classes: int, channels: Sequence[int] = CHANNELS, dropout: float = 0.1,
                 grid: int = GRID, canvas: int = CANVAS):
        super().__init__()
        self.grid, self.canvas = int(grid), int(canvas)
        self.neck = LearnedInterpolateToPyramidal([dim] * 4)
        self.decoder = UNetDecoder(self.neck.embedding_dim, list(channels))
        self.head = SegmentationHead(self.decoder.out_channels, n_classes, dropout)

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        """terratorch's ReshapeTokensToImage (no cls token, one timestep): (B, h*w, D) -> (B, D, h, w), row-major."""
        b, n, d = x.shape
        if n != self.grid * self.grid:
            raise ValueError(f"{n} tokens are not a {self.grid} x {self.grid} grid")
        return x.transpose(1, 2).reshape(b, d, self.grid, self.grid)

    def forward(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        feats = self.neck([self.reshape(t) for t in tokens])
        mask = self.head(self.decoder(feats))
        if mask.shape[-2:] != (self.canvas, self.canvas):
            mask = F.interpolate(mask, size=(self.canvas, self.canvas), mode="bilinear")
        return mask


# ------------------------------------------------------------------------------------------ the model
try:
    from transformers.utils import ModelOutput
except Exception:                                    # pragma: no cover
    ModelOutput = dict

from dataclasses import dataclass


@dataclass
class SegHeadOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None       # (B, C, native, native): the pooled logits


class SegHeadModel(nn.Module):
    """The LoRA'd decoder (its body only: no LM head) and the U-Net head, as one module for HF's Trainer."""

    accepts_loss_kwargs = False

    def __init__(self, lm: nn.Module, n_classes: int, ignore: Optional[int], native: int,
                 layers: Sequence[int] = LAYERS, copy_start: int = COPY_START, channels: Sequence[int] = CHANNELS,
                 dropout: float = 0.1, grid: int = GRID, canvas: int = CANVAS):
        super().__init__()
        self.lm = lm
        self.config = lm.config                       # HF's Trainer reads model.config in places
        self.n_classes, self.ignore, self.native = int(n_classes), ignore, int(native)
        self.layers, self.copy_start, self.n_tok = [int(i) for i in layers], int(copy_start), int(grid) * int(grid)
        n_layers = int(lm.config.num_hidden_layers)
        if not all(0 <= i <= n_layers for i in self.layers) or len(self.layers) != 4:
            raise ValueError(f"layers {self.layers}: four hidden-state indices in 0..{n_layers}")
        self.head = UNetHead(int(lm.config.hidden_size), self.n_classes, channels, dropout, grid, canvas)
        self.eval_cm: Optional[torch.Tensor] = None

    def features(self, input_ids, attention_mask=None, position_ids=None) -> List[torch.Tensor]:
        out = self.lm.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                            output_hidden_states=True, use_cache=False)
        a, b = self.copy_start, self.copy_start + self.n_tok
        if input_ids.shape[1] < b:
            raise ValueError(f"a sequence of {input_ids.shape[1]} tokens has no second copy at {a}..{b - 1}")
        return [out.hidden_states[i][:, a:b] for i in self.layers]

    def pooled_logits(self, input_ids, attention_mask=None, position_ids=None) -> torch.Tensor:
        """(B, C, native, native) fp32: the head's canvas logits pooled to the native grid (decode.to_native)."""
        from eo.maps import decode as D
        logits = self.head(self.features(input_ids, attention_mask, position_ids))
        dev = logits.device.type
        with torch.autocast(device_type=dev, enabled=False):
            return D.to_native(logits.float(), self.native)

    def forward(self, input_ids, attention_mask=None, position_ids=None, seg_labels=None, modality_ids=None):
        nat = self.pooled_logits(input_ids, attention_mask, position_ids)
        loss = None
        if seg_labels is not None:
            with torch.autocast(device_type=nat.device.type, enabled=False):
                loss = F.cross_entropy(nat, seg_labels.long(),
                                       ignore_index=self.ignore if self.ignore is not None else -100)
            if not self.training:                    # eval: the views' confusion matrix, read by SegHeadTrainer
                self._accumulate(nat.detach(), seg_labels)
        return SegHeadOutput(loss=loss, logits=nat)

    @torch.no_grad()
    def _accumulate(self, nat: torch.Tensor, gt: torch.Tensor) -> None:
        n = self.n_classes
        s = nat.clone()
        if self.ignore is not None:
            s[:, self.ignore] = -1e9
        pr = s.argmax(1)
        g = gt.long()
        v = g != self.ignore if self.ignore is not None else torch.ones_like(g, dtype=torch.bool)
        cm = torch.bincount((g[v] * n + pr[v]).flatten(), minlength=n * n).view(n, n).cpu()
        self.eval_cm = cm if self.eval_cm is None else self.eval_cm + cm


# ------------------------------------------------------------------------------------------ wiring
def crop_native(task: str) -> int:
    """The native size of one view's crop: PASTIS's 128-px patch, CloudSen12's 224-px crop (its tile is 512)."""
    from eo.maps import palette as P, tiling
    return int(tiling.plan(task, P.TASKS[task].native, "crop224")[0][2])


def wrap_model(model: nn.Module, cfg, finetune_report: Optional[Dict] = None) -> tuple:
    """pretrain.py's hook (seg_head.enable), after the fine-tune init applied LoRA: freeze the map rows, add the head.
    Returns (the wrapped model, the head's report). Given the fine-tune init's report, it records the head under
    "seg_head" and sets the report's trainable and total counts to the wrapped model's."""
    from eo.finetune.lora import SplitEmbedding
    from eo.maps import palette as P
    sh = cfg.seg_head
    if str(cfg.finetune.regime) != "lora":
        raise ValueError(f"Route C trains LoRA adapters and the head; finetune.regime is {cfg.finetune.regime!r}")
    emb = model.model.embed_tokens
    if not isinstance(emb, SplitEmbedding):
        raise RuntimeError("Route C expects the LoRA'd model's split vocabulary (eo/finetune/lora.apply_lora)")
    emb.new.requires_grad_(False)                    # no map token is written: the map rows would get no gradient
    task = str(cfg.seg.task)
    t = P.TASKS[task]
    wrapped = SegHeadModel(model, t.n_classes, t.ignore, crop_native(task), layers=list(sh.get("layers", LAYERS)),
                           copy_start=int(sh.get("copy_start", COPY_START)), channels=list(sh.get("channels", CHANNELS)),
                           dropout=float(sh.get("dropout", 0.1)))
    head_n = int(sum(p.numel() for p in wrapped.head.parameters()))
    lm_n = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    rep = {"route": "C", "task": task, "n_classes": t.n_classes, "ignore": t.ignore,
           "native": crop_native(task), "layers": wrapped.layers, "copy_start": wrapped.copy_start,
           "channels": list(sh.get("channels", CHANNELS)), "dropout": float(sh.get("dropout", 0.1)),
           "head_lr": float(sh.lr), "head_params": head_n, "lm_trainable_params": lm_n,
           "map_rows": "frozen", "trainable_params": head_n + lm_n}
    if finetune_report is not None:
        finetune_report["seg_head"] = rep
        finetune_report["trainable_params"] = int(sum(p.numel() for p in wrapped.parameters() if p.requires_grad))
        finetune_report["total_params"] = int(sum(p.numel() for p in wrapped.parameters()))
    return wrapped, rep


def head_param_names(model: nn.Module) -> List[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]


try:
    from util.trainer_pt import EOTrainer as _Base
except Exception:                                    # pragma: no cover (a2seg imports this module for the gate)
    _Base = object


class SegHeadTrainer(_Base):
    """EOTrainer with the head in its own parameter group (seg_head.lr), and the views' mIoU logged at each eval."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        model = self.model
        head_lr = float(self.cfg.seg_head.lr)
        decay = set(self.get_decay_parameter_names(model))
        groups = []
        for is_head, lr in ((False, float(self.args.learning_rate)), (True, head_lr)):
            for dec in (True, False):
                ps = [p for n, p in model.named_parameters()
                      if p.requires_grad and n.startswith("head.") == is_head and (n in decay) == dec]
                if ps:
                    groups.append({"params": ps, "lr": lr, "weight_decay": self.args.weight_decay if dec else 0.0,
                                   "group": "head" if is_head else "adapters"})
        opt_cls, kwargs = self.get_optimizer_cls_and_kwargs(self.args, model)
        kwargs = {k: v for k, v in kwargs.items() if k != "lr"}
        self.optimizer = opt_cls(groups, lr=float(self.args.learning_rate), **kwargs)
        return self.optimizer

    def evaluate(self, *args, **kwargs):
        m = self.model
        m.eval_cm = None
        metrics = super().evaluate(*args, **kwargs)
        if m.eval_cm is not None:
            cm = m.eval_cm.double().numpy()
            from eo.eval.seg_routes import miou_of
            v = miou_of(cm, m.ignore)
            metrics["eval_miou_views"] = v
            self.log({"eval_miou_views": v, "eval_pixel_acc_views": float(cm.trace() / max(cm.sum(), 1))})
        return metrics


# ------------------------------------------------------------------------------------------ evaluation
def build_head_model(config_name: str, checkpoint: Optional[str], device: str = "cuda",
                     overrides: Optional[Sequence[str]] = None):
    """A Route C fine-tune for evaluation, built as training built it: compose, build at the map vocabulary, apply LoRA,
    wrap, then strict-load the checkpoint's whole state (the trainer saves the wrapper's state dict)."""
    from pathlib import Path

    from eo.finetune import extend as X
    from eo.finetune import lora as LR
    from eo.finetune.init import _load_state, _weights, frozen_rows, strict_load, verify_extension
    from eo.generate.conditional import build_model
    model, cfg = build_model(config_name, None, device="cpu", overrides=overrides)
    ft = cfg.finetune
    if checkpoint is None:                           # the base itself under an untrained head (the gate's Z21c)
        verify_extension(Path(str(ft.init_from)), int(model.config.vocab_size))
        strict_load(model, Path(str(ft.init_from)))
        X.assert_tied(model)
    LR.apply_lora(model, int(ft.lora.r), float(ft.lora.alpha), frozen_rows(cfg))
    wrapped, _ = wrap_model(model, cfg)
    if checkpoint is not None:
        state = _load_state(_weights(Path(checkpoint)))
        missing, unexpected = wrapped.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"partial load from {checkpoint}: missing {list(missing)[:5]} ({len(missing)}), "
                               f"unexpected {list(unexpected)[:5]} ({len(unexpected)})")
    wrapped.to(device).eval()
    return wrapped, cfg


@torch.no_grad()
def predict_views(model: SegHeadModel, ds, views: Sequence[int], variant: str, batch: int, device,
                  autocast=None, log_every: int = 0) -> Dict:
    """Route C's maps: every sample of these views (whole samples, their crops in tiling order) -> (S, native, native)
    uint8 predictions. Per view: the pooled logits, log-softmax; CloudSen12's crops stitched by tiling.stitch (averaged);
    argmax with void masked (decode.apply_rule nearest)."""
    import time

    import numpy as np

    from eo.maps import decode as D, palette as P, tiling
    t = P.TASKS[ds.spec.task]
    crops = tiling.plan(ds.spec.task, t.native, "crop224")
    nc = len(crops)
    views = list(views)
    if len(views) % nc:
        raise ValueError(f"{len(views)} views are not whole samples of {nc} crops")
    preds, t0 = [], time.time()
    ctx = torch.autocast("cuda", dtype=autocast) if autocast is not None else _Null()
    per = max(1, batch // nc) * nc
    for s in range(0, len(views), per):
        vs = views[s:s + per]
        items = [ds.head_item(int(v), variant) for v in vs]
        ids = torch.stack([it["input_ids"] for it in items]).to(device)
        pos = torch.stack([it["position_ids"] for it in items]).to(device)
        am = torch.stack([it["attention_mask"] for it in items]).to(device)
        with ctx:
            nat = model.pooled_logits(ids, am, pos)
        lp = F.log_softmax(nat.float(), dim=1)
        sc = tiling.stitch(lp, crops, t.native) if nc > 1 else lp
        preds.append(D.apply_rule(sc, t.ignore, {"rule": "nearest"}, "logits").to(torch.uint8).cpu().numpy())
        if log_every and (s // per) % log_every == 0:
            print(f"  {s + len(vs)}/{len(views)} views ({time.time() - t0:.0f} s)", flush=True)
    return {"pred": np.concatenate(preds), "seconds": time.time() - t0}


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
