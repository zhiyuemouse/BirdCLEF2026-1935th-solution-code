import torch
import random
from functools import lru_cache
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torch.nn.functional as F

@lru_cache(maxsize=2)
def cached_audio_read(path):
    return sf.read(path)

class Bird2026Dataset_1(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.fs = 32000 # 采样率
        self.target_len = self.fs * 5
        # 梅尔频谱转换器
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.fs,
            n_fft=2048,        # 窗口大小，决定频率分辨率
            hop_length=512,    # 步长，决定时间分辨率 (32000/512 ≈ 62帧/秒)
            n_mels=128,        # 梅尔滤波器组的数量，即最终图像的高度
            f_min=20,          # 最低频率 (滤除超低频风噪)
            f_max=16000        # 最高频率 (奈奎斯特频率)
        )
        
        # 将能量转换为分贝 (对数尺度，增强微弱的鸟叫声特征)
        self.db_transform = T.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        data, samplerate = sf.read(row.file_path)
        data_len = len(data)
        if data_len < self.target_len: # 安全垫：如果音频本身比 5 秒短，先在末尾补零 (Padding)
            data = np.pad(data, (0, self.target_len - data_len), mode='constant') # np.pad 在 numpy 层面补齐
            data_len = len(data)
        if row.end == -1: # 短音频
            random_end = data_len - (self.fs * 5)
            audio_start = random.randint(0, random_end)
            audio_end = audio_start + (self.fs * 5)
        else:
            audio_start = int(float(row.start.split(":")[-1])) * self.fs
            audio_end = audio_start + (self.fs * 5)
            if audio_end > data_len: # 长音频尾部截断保护
                audio_start = data_len - self.target_len
                audio_end = data_len

        audio = data[audio_start: audio_end]
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        mel = self.mel_transform(audio_tensor) # 形状变为 (n_mels, time_frames)
        mel = self.db_transform(mel)           # 转换为分贝标度
        
        mel = mel.unsqueeze(0) # 增加一个 Channel 维度 (1, 128, 313)
        mel = torch.cat([mel] * 3, dim=0) # (3, 128, 313),方便使用 pretrained timm model
        
        label = np.array(row["1161364": "yeofly1"]).astype(np.float32)
        label = torch.from_numpy(label)

        return mel, label

class Bird2026Dataset_1_1_4(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.fs = 32000 # 采样率
        self.target_len = self.fs * 5
        self.db_transform = T.AmplitudeToDB(top_db=80)
        
        # 1. 顶部主图 (标准参数)
        self.mel_main = T.MelSpectrogram(
            sample_rate=self.fs, n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=16000)
            
        # 2. 左下部分 (高时间分辨率，捕捉快速急促的鸟叫，如啄木鸟)
        self.mel_time_res = T.MelSpectrogram(
            sample_rate=self.fs, n_fft=1024, hop_length=256, n_mels=128, f_min=20, f_max=16000)

        # 3. 右下2x2区块_左上 (高频率分辨率，捕捉音高的微小变化)
        self.mel_freq_res = T.MelSpectrogram(
            sample_rate=self.fs, n_fft=4096, hop_length=1024, n_mels=64, f_min=20, f_max=16000)
            
        # 4. 右下2x2区块_右上 (专注低频区，滤除高频虫鸣)
        self.mel_low_freq = T.MelSpectrogram(
            sample_rate=self.fs, n_fft=2048, hop_length=512, n_mels=64, f_min=20, f_max=4000)
            
        # 5. 右下2x2区块_左下 (专注高频区，专注某些特定高音昆虫或鸟类)
        self.mel_high_freq = T.MelSpectrogram(
            sample_rate=self.fs, n_fft=2048, hop_length=512, n_mels=64, f_min=4000, f_max=16000)
            
        # 6. 右下2x2区块_右下 (极端小窗口，极限时间感知)
        self.mel_extreme = T.MelSpectrogram(
            sample_rate=self.fs, n_fft=512, hop_length=128, n_mels=64, f_min=20, f_max=16000)

    def __len__(self):
        return len(self.df)

    def _get_mel_and_resize(self, audio_tensor, transform, target_size):
        """辅助函数：生成梅尔频谱并强制缩放至目标尺寸"""
        mel = transform(audio_tensor)
        mel = self.db_transform(mel)
        # interpolate 需要 4D tensor: (batch, channel, H, W)
        mel = mel.unsqueeze(0).unsqueeze(0) 
        # 双线性插值缩放
        mel = F.interpolate(mel, size=target_size, mode='bilinear', align_corners=False)
        # 挤压回 (channel, H, W)，这里 channel 为 1
        return mel.squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        data, samplerate = cached_audio_read(row.file_path)
        data_len = len(data)
        
        # 安全垫：音频过短补零
        if data_len < self.target_len: 
            data = np.pad(data, (0, self.target_len - data_len), mode='constant')
            data_len = len(data)
            
        # 随机裁剪或按标签截取
        if row.end == -1: 
            random_end = data_len - self.target_len
            audio_start = random.randint(0, random_end)
            audio_end = audio_start + self.target_len
        else:
            audio_start = int(float(row.start.split(":")[-1])) * self.fs
            audio_end = audio_start + self.target_len
            if audio_end > data_len:
                audio_start = data_len - self.target_len
                audio_end = data_len

        audio = data[audio_start: audio_end]
        audio_tensor = torch.from_numpy(audio.astype(np.float32))

        # ==================== 核心逻辑：生成多尺度特征并拼图 ====================
        
        # 1. 顶部大图: (1, 128, 312)
        top_mel = self._get_mel_and_resize(audio_tensor, self.mel_main, (128, 312))
        
        # 2. 左下方图: (1, 128, 156)
        bot_left = self._get_mel_and_resize(audio_tensor, self.mel_time_res, (128, 156))
        
        # 3. 右下方的 4 个小图: 每个 (1, 64, 78)
        br_1 = self._get_mel_and_resize(audio_tensor, self.mel_freq_res, (64, 78))
        br_2 = self._get_mel_and_resize(audio_tensor, self.mel_low_freq, (64, 78))
        br_3 = self._get_mel_and_resize(audio_tensor, self.mel_high_freq, (64, 78))
        br_4 = self._get_mel_and_resize(audio_tensor, self.mel_extreme, (64, 78))
        
        # 拼装右下角 2x2 网格
        # 行拼接 (沿着宽 W 的维度 dim=2)
        br_row1 = torch.cat([br_1, br_2], dim=2) # -> (1, 64, 156)
        br_row2 = torch.cat([br_3, br_4], dim=2) # -> (1, 64, 156)
        # 列拼接 (沿着高 H 的维度 dim=1)
        bot_right = torch.cat([br_row1, br_row2], dim=1) # -> (1, 128, 156)
        
        # 拼装整个下半部分 (沿着宽 W 的维度 dim=2)
        bottom_mel = torch.cat([bot_left, bot_right], dim=2) # -> (1, 128, 312)
        
        # 最终的终极拼图 (沿着高 H 的维度 dim=1)
        final_mel = torch.cat([top_mel, bottom_mel], dim=1) # -> (1, 256, 312)
        
        # =========================================================================

        # 转换为 3 通道以适配预训练 CV 模型 (3, 256, 312)
        final_mel = final_mel.repeat(3, 1, 1)
        mel_min = final_mel.min()
        mel_max = final_mel.max()
        final_mel = (final_mel - mel_min) / (mel_max - mel_min + 1e-6) # Min-Max 归一化，将其映射到 0.0 ~ 1.0 之间
        
        # 标签处理
        label = np.array(row["1161364": "yeofly1"]).astype(np.float32)
        label = torch.from_numpy(label)

        return final_mel, label