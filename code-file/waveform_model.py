import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dropout=0.0):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SincConv1D(nn.Module):
    """Learnable band-pass filters inspired by SincNet.

    This keeps the first raw-audio layer constrained to frequency-selective
    filters instead of asking a free Conv1d to rediscover a filterbank from
    limited labeled data.
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        stride,
        sample_rate=32000,
        min_low_hz=50.0,
        min_band_hz=50.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SincConv1D expects an odd kernel_size")
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.sample_rate = float(sample_rate)
        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(min_band_hz)

        max_hz = self.sample_rate / 2.0 - (self.min_low_hz + self.min_band_hz)
        hz = torch.linspace(self.min_low_hz, max_hz, self.out_channels + 1)
        self.low_hz_ = nn.Parameter(hz[:-1].clone().view(-1, 1))
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).clone().view(-1, 1))

        n = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1).float()
        self.register_buffer("n", n.view(1, -1))
        self.register_buffer("window", torch.hamming_window(self.kernel_size, periodic=False).view(1, -1))

    @staticmethod
    def _sinc(x):
        return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(x) / x)

    def forward(self, x):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            min=self.min_low_hz + self.min_band_hz,
            max=self.sample_rate / 2.0 - 1.0,
        )
        n = self.n.to(dtype=x.dtype, device=x.device)
        window = self.window.to(dtype=x.dtype, device=x.device)
        low = low.to(dtype=x.dtype, device=x.device)
        high = high.to(dtype=x.dtype, device=x.device)

        low_norm = low / self.sample_rate
        high_norm = high / self.sample_rate
        low_arg = 2.0 * torch.pi * low_norm * n
        high_arg = 2.0 * torch.pi * high_norm * n
        band_pass = 2.0 * high_norm * self._sinc(high_arg) - 2.0 * low_norm * self._sinc(low_arg)
        band_pass = band_pass * window
        band_pass = band_pass / (band_pass.abs().sum(dim=1, keepdim=True) + 1e-8)
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=self.stride, padding=self.kernel_size // 2)


class RawAudioTokenizer(nn.Module):
    """
    Input:
        x: [B,160000]

    Output:
        tokens: [B,num_tokens,768]
    """
    def __init__(self, embed_dim=768, num_tokens=32):
        super().__init__()
        if num_tokens == 16:
            fourth_stride = 8
        elif num_tokens == 32:
            fourth_stride = 4
        elif num_tokens == 64:
            fourth_stride = 2
        else:
            raise ValueError(f"RawAudioTokenizer supports num_tokens 16, 32, or 64, got {num_tokens}")

        self.num_tokens = int(num_tokens)

        self.conv = nn.Sequential(
            ConvBlock1D(1, 64,   kernel_size=401, stride=10, dropout=0.0),
            ConvBlock1D(64, 128, kernel_size=101, stride=5,  dropout=0.0),
            ConvBlock1D(128, 256, kernel_size=51, stride=5,  dropout=0.05),
            ConvBlock1D(256, 384, kernel_size=25, stride=fourth_stride,  dropout=0.05),
            ConvBlock1D(384, embed_dim, kernel_size=17, stride=5, dropout=0.05),
        )

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B,160000]
        x = x.unsqueeze(1)              # [B,1,160000]
        x = self.conv(x)                # [B,768,num_tokens]
        if x.shape[-1] != self.num_tokens:
            raise RuntimeError(f"Expected {self.num_tokens} waveform tokens, got {x.shape[-1]}")
        x = x.transpose(1, 2)           # [B,num_tokens,768]
        x = self.out_norm(x)            # [B,num_tokens,768]
        return x


class SincRawAudioTokenizer(nn.Module):
    """
    Input:
        x: [B,160000]

    Output:
        tokens: [B,num_tokens,768]
    """
    def __init__(self, embed_dim=768, num_tokens=32, sample_rate=32000):
        super().__init__()
        if num_tokens == 16:
            fourth_stride = 8
        elif num_tokens == 32:
            fourth_stride = 4
        elif num_tokens == 64:
            fourth_stride = 2
        else:
            raise ValueError(f"SincRawAudioTokenizer supports num_tokens 16, 32, or 64, got {num_tokens}")

        self.num_tokens = int(num_tokens)

        self.sinc = nn.Sequential(
            SincConv1D(64, kernel_size=401, stride=10, sample_rate=sample_rate),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            ConvBlock1D(64, 128, kernel_size=101, stride=5, dropout=0.0),
            ConvBlock1D(128, 256, kernel_size=51, stride=5, dropout=0.05),
            ConvBlock1D(256, 384, kernel_size=25, stride=fourth_stride, dropout=0.05),
            ConvBlock1D(384, embed_dim, kernel_size=17, stride=5, dropout=0.05),
        )

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B,160000]
        x = x.unsqueeze(1)              # [B,1,160000]
        x = self.sinc(x)                # [B,64,16000]
        x = self.conv(x)                # [B,768,num_tokens]
        if x.shape[-1] != self.num_tokens:
            raise RuntimeError(f"Expected {self.num_tokens} waveform tokens, got {x.shape[-1]}")
        x = x.transpose(1, 2)           # [B,num_tokens,768]
        x = self.out_norm(x)            # [B,num_tokens,768]
        return x
    
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x):
        # x: [B,N,D]
        score = self.attn(x)               # [B,N,1]
        weight = torch.softmax(score, dim=1)
        pooled = (x * weight).sum(dim=1)   # [B,D]
        return pooled

class RawWaveTransformerModel(nn.Module):
    def __init__(
        self,
        num_classes=234,
        embed_dim=768,
        depth=4,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        num_tokens=32,
        tokenizer_type="conv_stack",
    ):
        super().__init__()

        if tokenizer_type == "sinc_stack":
            self.tokenizer = SincRawAudioTokenizer(embed_dim=embed_dim, num_tokens=num_tokens)
        elif tokenizer_type == "conv_stack":
            self.tokenizer = RawAudioTokenizer(embed_dim=embed_dim, num_tokens=num_tokens)
        else:
            raise ValueError(f"Unsupported tokenizer_type for RawWaveTransformerModel: {tokenizer_type}")

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.attn_pool = AttentionPooling(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        # x: [B,160000]

        x = self.tokenizer(x)       # [B,32,768]
        x = x + self.pos_embed      # [B,32,768]

        x = self.encoder(x)         # [B,32,768]

        x_mean = x.mean(dim=1)      # [B,768]
        x_max = x.amax(dim=1)       # [B,768]
        x_attn = self.attn_pool(x)  # [B,768]

        feat = torch.cat([x_mean, x_max, x_attn], dim=-1)  # [B,2304]

        return feat

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.head(feat)    # [B,234]
        return logits

class TemporalConvMixer(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.conv = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                bias=False,
            ),
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B,N,D]
        shortcut = x

        x = self.norm(x)
        x = x.transpose(1, 2)   # [B,D,N]
        x = self.conv(x)
        x = x.transpose(1, 2)   # [B,N,D]

        return shortcut + x

class RawWaveTransformerMixerModel(nn.Module):
    def __init__(
        self,
        num_classes=234,
        embed_dim=768,
        depth=4,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        num_tokens=32,
        tokenizer_type="conv_stack",
    ):
        super().__init__()

        if tokenizer_type == "sinc_stack":
            self.tokenizer = SincRawAudioTokenizer(embed_dim=embed_dim, num_tokens=num_tokens)
        elif tokenizer_type == "conv_stack":
            self.tokenizer = RawAudioTokenizer(embed_dim=embed_dim, num_tokens=num_tokens)
        else:
            raise ValueError(f"Unsupported tokenizer_type for RawWaveTransformerMixerModel: {tokenizer_type}")

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        self.local_mixer = nn.Sequential(
            TemporalConvMixer(embed_dim, kernel_size=5, dropout=0.1),
            TemporalConvMixer(embed_dim, kernel_size=5, dropout=0.1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.attn_pool = AttentionPooling(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        # x: [B,160000]

        x = self.tokenizer(x)       # [B,32,768]
        x = x + self.pos_embed      # [B,32,768]
        x = self.local_mixer(x)     # [B,32,768]

        x = self.encoder(x)         # [B,32,768]

        x_mean = x.mean(dim=1)      # [B,768]
        x_max = x.amax(dim=1)       # [B,768]
        x_attn = self.attn_pool(x)  # [B,768]

        feat = torch.cat([x_mean, x_max, x_attn], dim=-1)  # [B,2304]

        return feat

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.head(feat)    # [B,234]
        return logits
    
if __name__ == "__main__":
    model = RawWaveTransformerModel(num_classes=234)

    waveform = torch.randn(8, 160000)
    logits = model(waveform)

    print(logits.shape)
