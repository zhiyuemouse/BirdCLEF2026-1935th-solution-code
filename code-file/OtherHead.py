import torch
import torch.nn as nn
import math

class PerchFusionHead(nn.Module):
    def __init__(self, num_classes=234, selected_label_dim=1000):
        super().__init__()

        input_dim = 1536 + 1536 + 1536 + selected_label_dim

        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, embedding, spatial, selected_label_logits):
        # embedding: [B,1536]
        # spatial: [B,16,4,1536]
        # selected_label_logits: [B,K]

        spatial_mean = spatial.mean(dim=(1, 2))  # [B,1536]
        spatial_max = spatial.amax(dim=(1, 2))   # [B,1536]

        x = torch.cat([
            embedding,
            spatial_mean,
            spatial_max,
            selected_label_logits,
        ], dim=-1)

        return self.head(x)
    
class PerchStrongHead(nn.Module):
    def __init__(self, in_features=1536, hidden=768, num_classes=234):
        super().__init__()

        self.token_feat = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.token_classifier = nn.Linear(hidden, num_classes)
        self.token_attention = nn.Linear(hidden, num_classes)

        self.global_head = nn.Sequential(
            nn.LayerNorm(in_features * 2),
            nn.Linear(in_features * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

        self.final_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, spatial):
        # spatial: [B,16,4,1536]
        B = spatial.shape[0]

        # local token path
        x = spatial.reshape(B, 64, 1536)        # [B,64,1536]
        h = self.token_feat(x)                  # [B,64,hidden]

        token_logits = self.token_classifier(h) # [B,64,234]
        attn_score = self.token_attention(h)    # [B,64,234]
        attn_weight = torch.softmax(attn_score, dim=1)

        mil_logits = (token_logits * attn_weight).sum(dim=1)  # [B,234]

        # global pooled path
        spatial_mean = spatial.mean(dim=(1, 2)) # [B,1536]
        spatial_max = spatial.amax(dim=(1, 2))  # [B,1536]
        global_feat = torch.cat([spatial_mean, spatial_max], dim=-1)

        global_logits = self.global_head(global_feat)          # [B,234]

        # combine
        logits = mil_logits + self.final_scale * global_logits

        return logits