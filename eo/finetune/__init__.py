"""Phase 6 fine-tuning machinery (`.venv`).

    extend.py   grow a pretrained checkpoint's vocabulary by the 21 BEN ids, re-tie, and
                prove every pretraining logit is unchanged (D6.4)
    lora.py     hand-rolled LoRA; the new vocabulary rows are a SEPARATE parameter (§4.7)
    init.py     the init-from-checkpoint path pretrain.py calls: strict load of an extended
                checkpoint, then the regime (full / LoRA). Not a resume (§4.9).
"""
