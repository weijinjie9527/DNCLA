import torch
import torch.nn as nn


class CLAModel(nn.Module):
    """消融实验：去除 Attention，仅保留 CNN + BiLSTM"""

    def __init__(self):
        super(CLAModel, self).__init__()

        # 分支 A: NCP 多尺度 CNN
        self.ncp_m_conv1 = nn.Sequential(nn.Conv1d(9, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32))
        self.ncp_m_conv2 = nn.Sequential(nn.Conv1d(9, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64))
        self.ncp_m_conv3 = nn.Sequential(nn.Conv1d(9, 128, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(128))
        self.ncp_pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.ncp_post_conv = nn.Sequential(nn.Conv1d(224, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128))
        self.ncp_lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)

        # 分支 B: DPCP 多尺度 CNN
        self.oh_m_conv1 = nn.Sequential(nn.Conv1d(15, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32))
        self.oh_m_conv2 = nn.Sequential(nn.Conv1d(15, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64))
        self.oh_m_conv3 = nn.Sequential(nn.Conv1d(15, 128, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(128))
        self.oh_pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.oh_post_conv = nn.Sequential(nn.Conv1d(224, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128))
        self.oh_lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)

        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, ncp, dpcp):
        # NCP 分支：CNN → BiLSTM → Pool (无 Attention)
        a1, a2, a3 = self.ncp_m_conv1(ncp), self.ncp_m_conv2(ncp), self.ncp_m_conv3(ncp)
        na = self.ncp_post_conv(self.ncp_pool(torch.cat((a1, a2, a3), dim=1)))
        na_l, _ = self.ncp_lstm(na.permute(0, 2, 1))
        na_v = self.adaptive_pool(na_l.permute(0, 2, 1)).squeeze(-1)

        # DPCP 分支：CNN → BiLSTM → Pool (无 Attention)
        b1, b2, b3 = self.oh_m_conv1(dpcp), self.oh_m_conv2(dpcp), self.oh_m_conv3(dpcp)
        ob = self.oh_post_conv(self.oh_pool(torch.cat((b1, b2, b3), dim=1)))
        ob_l, _ = self.oh_lstm(ob.permute(0, 2, 1))
        ob_v = self.adaptive_pool(ob_l.permute(0, 2, 1)).squeeze(-1)

        return self.classifier(torch.cat((na_v, ob_v), dim=1))
