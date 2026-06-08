#########################################
# Import
#########################################
import os
import warnings # 避免一些可以忽略的报错
warnings.filterwarnings('ignore')

import random
import pandas as pd
import numpy as np
from functools import lru_cache
import soundfile as sf
import glob
from tqdm import tqdm

import timm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T

#########################################
# CONFIG
#########################################
class CONFIG:
    is_debug = False
    seed = 308
    n_workers = os.cpu_count() // 2
    test_audio_path = "/kaggle/input/competitions/birdclef-2026/test_soundscapes"
    if is_debug:
        test_audio_path = "/kaggle/input/competitions/birdclef-2026/train_soundscapes"
    
    model_name = "convnext_atto.d2_in1k"
    '''
    tf_efficientnet_b0.ns_jft_in1k
    convnext_atto.d2_in1k
    '''
    ckpt_path = "/kaggle/input/models/zhiyue666/20260407-atto-5fold-cv7681/pytorch/default/1"
    test_batch_size = 256
    head_out = 234

#########################################
# Seeding
#########################################
def set_seed(seed=308):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
set_seed(CONFIG.seed)

#########################################
# Data Progress
#########################################
test_dir = CONFIG.test_audio_path

test_paths = sorted(os.listdir(test_dir))
if CONFIG.is_debug:
    test_paths = test_paths[: 10]

all_names = []
for path in test_paths:
    if path[-4:] != ".ogg":
        continue
    name = path.split(".")[0]
    all_names.append(name + "_5")
    all_names.append(name + "_10")
    all_names.append(name + "_15")
    all_names.append(name + "_20")
    all_names.append(name + "_25")
    all_names.append(name + "_30")
    all_names.append(name + "_35")
    all_names.append(name + "_40")
    all_names.append(name + "_45")
    all_names.append(name + "_50")
    all_names.append(name + "_55")
    all_names.append(name + "_60")

test = pd.DataFrame(all_names, columns=["row_id"])

#########################################
# Dataset and DataLoader
#########################################
@lru_cache(maxsize=2)
def cached_audio_read(path):
    return sf.read(path)

class Bird2026Dataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.fs = 32000 # 采样率
        self.target_len = self.fs * 5
        self.db_transform = T.AmplitudeToDB(top_db=80) # 将能量转换为分贝 (对数尺度，增强微弱的鸟叫声特征)
        # 梅尔频谱转换器
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        path = CONFIG.test_audio_path + "/" + row.row_id.rsplit('_', 1)[0] + ".ogg"
        data, samplerate = cached_audio_read(path) # 缓存读取
        data_len = len(data)
        if data_len < self.target_len: # 安全垫
            data = np.pad(data, (0, self.target_len - data_len), mode='constant')
            data_len = len(data)

        audio_start = (int(row.row_id.split("_")[-1]) - 5) * self.fs # 取出结尾秒，减5就是开始秒
        audio_end = audio_start + (self.fs * 5)
        if audio_end > data_len: # 长音频尾部截断保护
            audio_start = data_len - self.target_len
            audio_end = data_len

        audio = data[audio_start: audio_end]
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
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

        return final_mel, row.row_id
    
def prepare_loaders(df):
    test_datasets = Bird2026Dataset(df=df)
    
    test_loader = DataLoader(test_datasets, batch_size=CONFIG.test_batch_size, num_workers=0, shuffle=False, pin_memory=True)
    
    return test_loader

#########################################
# Model
#########################################
class Bird2026Model(nn.Module):
    def __init__(self, model_name=CONFIG.model_name, is_pretrained=False):
        super(Bird2026Model, self).__init__()
        self.backbone = timm.create_model(model_name=model_name, 
                                          pretrained=is_pretrained)
        if "efficientnet" in CONFIG.model_name:
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif "convnext" in CONFIG.model_name:
            in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features // 2, CONFIG.head_out),
        )

    def forward(self, x):
        _tmp = self.backbone(x)
        output = self.head(_tmp)
        return output
    
model = Bird2026Model()
print(model)

#########################################
# Load Models
#########################################
models = []

paths = os.listdir(CONFIG.ckpt_path)
paths = sorted(paths, key=lambda x:x.split("_"[0]))
length = len(paths)

for i in range(length):
    model = Bird2026Model()
    model.load_state_dict(torch.load(os.path.join(CONFIG.ckpt_path, paths[i]), map_location="cpu"))
    model.eval()
    models.append(model)
    print(f"Fold_{paths[i]} Load Success!")

#########################################
# Infer Function
#########################################
def Infer(models, test_loader):
    y_preds = []
    bar = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for step, (audio, study_id) in bar:   
            outputs = 0
            for model in models:
                output = F.sigmoid(model(audio))
                outputs += output
            outputs = outputs / len(models)
            y_preds.append(outputs.detach().cpu().numpy())
            
    y_preds = np.concatenate(y_preds)
    return y_preds

#########################################
# Start Infer
#########################################
test_loader = prepare_loaders(test)

preds = Infer(models, test_loader)

train_csv = pd.read_csv("/kaggle/input/datasets/zhiyue666/bird2026-train-csv/train_5folds_sgkf.csv")
# 定义要排除的列
exclude_cols = ['file_path', 'start', 'end', 'group', 'proxy_label', 'fold']

# 获取剩余列的列表
test_cols = [col for col in train_csv.columns if col not in exclude_cols]
test[test_cols] = -1
test.loc[:, test_cols] = preds
print(test)
test.to_csv("submission.csv", index=False)
