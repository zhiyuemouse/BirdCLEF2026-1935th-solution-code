import torch
import torch.nn as nn

class LocalMambaBlock(nn.Module):
    """
    Lightweight Mamba-style block (Gated CNN) from the reference notebook.
    Efficiently mixes tokens with linear complexity.
    """
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # Depthwise conv mixes spatial information locally
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )

        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Tokens, Dim)
        shortcut = x

        x = self.norm(x)

        # Gating mechanism
        g = torch.sigmoid(self.gate(x))
        x = x * g

        # Spatial mixing via 1D Conv (requires transpose)
        x = x.transpose(1, 2)  # -> (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # -> (B, N, D)

        # Projection
        x = self.proj(x)
        x = self.drop(x)

        return shortcut + x

class PerchMambaHead(nn.Module):
    def __init__(self, in_features=1536, num_classes=234):
        super().__init__()

        self.fusion = nn.Sequential(
            LocalMambaBlock(in_features, kernel_size=5, dropout=0.1),
            LocalMambaBlock(in_features, kernel_size=5, dropout=0.1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features // 2, num_classes),
        )

    def forward(self, spatial_embedding):
        # spatial_embedding: [B,16,4,1536]

        x = spatial_embedding.mean(dim=2)  # [B,16,1536]

        x = self.fusion(x)                 # [B,16,1536]

        x = self.pool(x.transpose(1, 2))   # [B,1536,1]
        x = x.flatten(1)                   # [B,1536]

        logits = self.head(x)              # [B,234]

        return logits
    
class PerchMambaHeadMeanMax(nn.Module):
    def __init__(self, in_features=1536, num_classes=234):
        super().__init__()

        self.freq_proj = nn.Sequential(
            nn.Linear(in_features * 2, in_features),
            nn.LayerNorm(in_features),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )

        self.fusion = nn.Sequential(
            LocalMambaBlock(in_features, kernel_size=5, dropout=0.1),
            LocalMambaBlock(in_features, kernel_size=5, dropout=0.1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features // 2, num_classes),
        )

    def forward(self, spatial_embedding):
        # spatial_embedding: [B,16,4,1536]

        x_mean = spatial_embedding.mean(dim=2)       # [B,16,1536]
        x_max = spatial_embedding.max(dim=2).values  # [B,16,1536]

        x = torch.cat([x_mean, x_max], dim=-1)       # [B,16,3072]
        x = self.freq_proj(x)                       # [B,16,1536]

        x = self.fusion(x)                          # [B,16,1536]

        x = self.pool(x.transpose(1, 2)).flatten(1)  # [B,1536]

        logits = self.head(x)                       # [B,234]

        return logits
    

class PerchMambaHead64Tokens(nn.Module):
    def __init__(self, in_features=1536, num_classes=234):
        super().__init__()

        self.fusion = nn.Sequential(
            LocalMambaBlock(in_features, kernel_size=9, dropout=0.1),
            LocalMambaBlock(in_features, kernel_size=9, dropout=0.1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features // 2, num_classes),
        )

    def forward(self, spatial_embedding):
        # spatial_embedding: [B,16,4,1536]

        B = spatial_embedding.shape[0]

        x = spatial_embedding.reshape(B, 16 * 4, 1536)  # [B,64,1536]

        x = self.fusion(x)                              # [B,64,1536]

        x = self.pool(x.transpose(1, 2)).flatten(1)      # [B,1536]

        logits = self.head(x)                           # [B,234]

        return logits
    
class PerchMambaHead64Tokens_with_pos_embed(nn.Module):
    def __init__(self, in_features=1536, num_classes=234):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, 64, 1536))

        self.fusion = nn.Sequential(
            LocalMambaBlock(in_features, kernel_size=9, dropout=0.1),
            LocalMambaBlock(in_features, kernel_size=9, dropout=0.1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features // 2, num_classes),
        )

    def forward(self, spatial_embedding):
        # spatial_embedding: [B,16,4,1536]

        B = spatial_embedding.shape[0]

        x = spatial_embedding.reshape(B, 16 * 4, 1536)  # [B,64,1536]

        x = x + self.pos_embed                          # [B,64,1536]

        x = self.fusion(x)                              # [B,64,1536]

        x = self.pool(x.transpose(1, 2)).flatten(1)      # [B,1536]

        logits = self.head(x)                           # [B,234]

        return logits