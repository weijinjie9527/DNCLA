import torch
import torch.nn as nn


class CLAModel(nn.Module):
    """消融实验：去除CNN，仅保留 BiLSTM + Attention"""

    def __init__(self):
        super(CLAModel, self).__init__()

        # 分支 A: NCP 投影 (替代 CNN)
        self.ncp_proj = nn.Sequential(
            nn.Conv1d(9, 128, 1), nn.ReLU(), nn.BatchNorm1d(128)
        )
        self.ncp_lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.ncp_attn = nn.MultiheadAttention(256, 8, batch_first=True)

        # 分支 B: DPCP 投影 (替代 CNN)
        self.dpcp_proj = nn.Sequential(
            nn.Conv1d(15, 128, 1), nn.ReLU(), nn.BatchNorm1d(128)
        )
        self.dpcp_lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.dpcp_attn = nn.MultiheadAttention(256, 8, batch_first=True)

        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, ncp, dpcp):
        # NCP 分支：投影 → BiLSTM → Attention
        na = self.ncp_proj(ncp)
        na_l, _ = self.ncp_lstm(na.permute(0, 2, 1))
        na_at, _ = self.ncp_attn(na_l, na_l, na_l)
        na_v = self.adaptive_pool(na_at.permute(0, 2, 1)).squeeze(-1)

        # DPCP 分支：投影 → BiLSTM → Attention
        db = self.dpcp_proj(dpcp)
        db_l, _ = self.dpcp_lstm(db.permute(0, 2, 1))
        db_at, _ = self.dpcp_attn(db_l, db_l, db_l)
        db_v = self.adaptive_pool(db_at.permute(0, 2, 1)).squeeze(-1)

        return self.classifier(torch.cat((na_v, db_v), dim=1))
