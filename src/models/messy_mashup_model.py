import logging

import torch
import torch.nn as nn

from src.models.attention import StemAttentionPool
from src.models.crnn import TemporalCRNN
from src.models.fusion import FusionModule
from src.models.stem_branch import StemEncoder

logger = logging.getLogger(__name__)

STEM_KEYS = ["bass", "drums", "other", "vocals"]


class MultiBranchCRNN(nn.Module):
    """
    Parameters
    ----------
    num_classes     : number of output genres
    model_cfg       : dict matching model_config.yaml model section
    """

    def __init__(self, num_classes: int = 10, model_cfg: dict | None = None) -> None:
        super().__init__()

        if model_cfg is None:
            model_cfg = {}

        enc_cfg = model_cfg.get("stem_encoder", {})
        att_cfg = model_cfg.get("attention", {})
        fus_cfg = model_cfg.get("fusion", {})
        crnn_cfg = model_cfg.get("crnn", {})
        cls_cfg = model_cfg.get("classifier", {})

        channels = enc_cfg.get("channels", [32, 64, 128])
        embed_dim = att_cfg.get("embed_dim", 128)
        enc_dropout = enc_cfg.get("dropout", 0.2)

        # 4 independent stem encoders
        self.stem_encoders = nn.ModuleDict({
            stem: StemEncoder(channels=channels, embed_dim=embed_dim, dropout=enc_dropout)
            for stem in STEM_KEYS
        })

        # Separate mixture encoder
        self.mix_encoder = StemEncoder(channels=channels, embed_dim=embed_dim, dropout=enc_dropout)

        # Attention pooling over 4 stem embeddings
        self.attention = StemAttentionPool(embed_dim=embed_dim)

        # Feature fusion
        projected_dim = fus_cfg.get("projected_dim", 256)
        self.fusion = FusionModule(
            stem_embed_dim=embed_dim,
            mix_embed_dim=embed_dim,
            projected_dim=projected_dim,
        )

        # Temporal BiGRU
        crnn_hidden = crnn_cfg.get("hidden_size", 256)
        self.crnn = TemporalCRNN(
            input_size=projected_dim,
            hidden_size=crnn_hidden,
            num_layers=crnn_cfg.get("num_layers", 2),
            dropout=crnn_cfg.get("dropout", 0.3),
        )

        # Classifier head
        cls_hidden = cls_cfg.get("hidden_dim", 256)
        cls_dropout = cls_cfg.get("dropout", 0.4)
        self.classifier = nn.Sequential(
            nn.Linear(crnn_hidden, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, num_classes),
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MultiBranchCRNN initialised — {n_params:,} trainable parameters")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        batch : dict with keys bass/drums/other/vocals/mix,
                each (B, 1, n_mels, T)

        Returns dict with:
            logits        : (B, num_classes)
            stem_weights  : (B, 4)   — attention weights per stem
        """
        # Encode each stem
        stem_embeds = []
        for stem in STEM_KEYS:
            assert stem in batch, f"Missing stem in batch: {stem!r}"
            emb = self.stem_encoders[stem](batch[stem])   # (B, embed_dim)
            stem_embeds.append(emb)

        # Stack → (B, 4, embed_dim)
        stem_stack = torch.stack(stem_embeds, dim=1)

        # Attend over stems → (B, embed_dim)
        attended, stem_weights = self.attention(stem_stack)

        # Mix branch
        assert "mix" in batch, "Missing 'mix' key in batch"
        mix_emb = self.mix_encoder(batch["mix"])          # (B, embed_dim)

        # Fuse
        fused = self.fusion(attended, mix_emb)            # (B, projected_dim)

        # Temporal context
        temporal = self.crnn(fused)                        # (B, hidden_size)

        # Classify
        logits = self.classifier(temporal)                 # (B, num_classes)

        return {"logits": logits, "stem_weights": stem_weights}


def build_model(num_classes: int, model_cfg: dict | None = None) -> MultiBranchCRNN:
    return MultiBranchCRNN(num_classes=num_classes, model_cfg=model_cfg)