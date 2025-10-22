import torch
from torch import nn

class BoundarySensitiveAttention(nn.Module):  # BSA
    def __init__(self, channels, num_classes=2):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.enhancement = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.GELU())
        self.cls1 = nn.Conv2d(channels, num_classes, 1)
        self.cls2 = nn.Conv2d(channels, num_classes, 1)
        self.region_learner = nn.Sequential(nn.Conv2d(num_classes, 1, 1), nn.BatchNorm2d(1), nn.GELU())

    def forward(self, x1, x2):
        prob1 = torch.sigmoid(self.cls1(x1))
        prob2 = torch.sigmoid(self.cls2(x2))
        diff_map = torch.abs(prob1 - prob2)
        attn = self.sigmoid(self.region_learner(diff_map))
        return self.enhancement(attn * x1 + x1)


class HierarchicalReceptiveFieldFusion(nn.Module):  # HRFF
    def __init__(self, c_s, c_m, c_l, embed_dim, num_classes=2, drop_rate=0.2):
        super().__init__()
        self.conv_s = nn.Conv2d(c_s, embed_dim, 1)
        self.pool_s = nn.MaxPool2d(4)
        self.conv_m = nn.Conv2d(c_m, embed_dim, 1)
        self.pool_m = nn.MaxPool2d(2)
        self.conv_l = nn.Conv2d(c_l, embed_dim, 1)
        self.fusion = nn.Sequential(nn.Conv2d(embed_dim * 3, embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.GELU())
        self.dropout = nn.Dropout2d(drop_rate)
        self.bsa1 = BoundarySensitiveAttention(embed_dim, num_classes)
        self.bsa2 = BoundarySensitiveAttention(embed_dim, num_classes)

    def forward(self, features):
        x_s, x_m, x_l = features
        x_s_p = self.pool_s(self.conv_s(x_s))
        x_m_p = self.pool_m(self.conv_m(x_m))
        x_l_p = self.conv_l(x_l)
        fused = self.fusion(torch.cat([x_s_p, x_m_p, x_l_p], dim=1))
        edge1 = self.bsa1(x_m_p, x_s_p)
        edge2 = self.bsa2(x_l_p, x_m_p)
        return self.dropout(fused) + edge1 + edge2