import torch
import torch.nn as nn

class AttentionPoolingHead(nn.Module):
    def __init__(self, in_features=1536, num_classes=234):
        super().__init__()

        self.attn = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes),
        )

    def forward(self, spatial):
        # spatial: [B,16,4,1536]
        B = spatial.shape[0]
        x = spatial.reshape(B, 64, 1536)  # [B,64,1536]

        score = self.attn(x)              # [B,64,1]
        weight = torch.softmax(score, dim=1)
        pooled = (x * weight).sum(dim=1)  # [B,1536]

        return self.head(pooled)
    
class MultiHeadAttentionPoolingHead(nn.Module):
    def __init__(self, in_features=1536, num_classes=234, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

        self.attn = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.Tanh(),
            nn.Linear(256, num_heads),
        )

        self.proj = nn.Sequential(
            nn.LayerNorm(in_features * num_heads),
            nn.Linear(in_features * num_heads, in_features),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, spatial):
        # spatial: [B,16,4,1536]
        B = spatial.shape[0]
        x = spatial.reshape(B, 64, 1536)          # [B,64,1536]

        score = self.attn(x)                     # [B,64,H]
        weight = torch.softmax(score, dim=1)     # [B,64,H]

        x_expanded = x.unsqueeze(2)              # [B,64,1,1536]
        weight = weight.unsqueeze(-1)            # [B,64,H,1]

        pooled = (x_expanded * weight).sum(dim=1) # [B,H,1536]
        pooled = pooled.reshape(B, -1)            # [B,H*1536]

        feat = self.proj(pooled)                  # [B,1536]
        return self.classifier(feat)              # [B,234]
    
class PerchTransformerHead(nn.Module):
    def __init__(self, in_features=1536, num_classes=234, num_layers=1, num_heads=4):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, 64, in_features))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_features,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(in_features * 2),
            nn.Linear(in_features * 2, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes),
        )

    def forward(self, spatial):
        # spatial: [B,16,4,1536]
        B = spatial.shape[0]

        x = spatial.reshape(B, 64, 1536)  # [B,64,1536]
        x = x + self.pos_embed

        x = self.encoder(x)               # [B,64,1536]

        x_mean = x.mean(dim=1)            # [B,1536]
        x_max = x.amax(dim=1)             # [B,1536]
        x = torch.cat([x_mean, x_max], dim=-1)

        return self.head(x)
    

class SmallTransformerHead(nn.Module):
    def __init__(self, in_features=1536, hidden=512, num_classes=234):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden),
            nn.GELU(),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 64, hidden))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 4,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, spatial):
        B = spatial.shape[0]

        x = spatial.reshape(B, 64, 1536)  # [B,64,1536]
        x = self.input_proj(x)            # [B,64,512]
        x = x + self.pos_embed
        x = self.encoder(x)               # [B,64,512]

        x_mean = x.mean(dim=1)
        x_max = x.amax(dim=1)

        x = torch.cat([x_mean, x_max], dim=-1)
        return self.head(x)