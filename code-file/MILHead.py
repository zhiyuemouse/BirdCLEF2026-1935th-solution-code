import torch
import torch.nn as nn
import math

class MILHead(nn.Module):
    def __init__(self, in_features=1536, num_classes=234):
        super().__init__()

        self.token_classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, num_classes),
        )

    def forward(self, spatial):
        # spatial: [B,16,4,1536]
        B = spatial.shape[0]
        x = spatial.reshape(B, 64, 1536)       # [B,64,1536]

        token_logits = self.token_classifier(x) # [B,64,234]

        # clip-level aggregation
        logits = token_logits.amax(dim=1)       # [B,234]

        return logits
    
class LogSumExpMILHead(nn.Module):
    def __init__(self, in_features=1536, num_classes=234):
        super().__init__()

        self.token_classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, num_classes),
        )

    def forward(self, spatial):
        B = spatial.shape[0]
        x = spatial.reshape(B, 64, 1536)        # [B,64,1536]

        token_logits = self.token_classifier(x) # [B,64,234]

        logits = torch.logsumexp(token_logits, dim=1) - math.log(x.shape[1])

        return logits
    
class AttentionMILHead(nn.Module):
    def __init__(self, in_features=1536, num_classes=234):
        super().__init__()

        self.feat = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 768),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(768, num_classes)
        self.attention = nn.Linear(768, num_classes)

    def forward(self, spatial):
        # spatial: [B,16,4,1536]
        B = spatial.shape[0]
        x = spatial.reshape(B, 64, 1536)      # [B,64,1536]

        h = self.feat(x)                      # [B,64,768]

        token_logits = self.classifier(h)     # [B,64,234]
        attn_score = self.attention(h)        # [B,64,234]

        attn_weight = torch.softmax(attn_score, dim=1)

        logits = (token_logits * attn_weight).sum(dim=1)  # [B,234]

        return logits