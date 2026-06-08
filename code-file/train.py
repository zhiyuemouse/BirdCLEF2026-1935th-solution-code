#########################################
# Import
#########################################
import os
import warnings # 避免一些可以忽略的报错
warnings.filterwarnings('ignore')
import sys
import random
import copy
import math
from tqdm import tqdm
import time
import gc
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import shutil

import pandas as pd
import numpy as np
import soundfile as sf

import timm
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler # 学习率调度器
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
import torchaudio.transforms as T

from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

#########################################
# CONFIG
#########################################
class CONFIG:
    is_debug = False
    seed = 308
    n_folds = 5
    n_workers = os.cpu_count() // 2

    train_csv = "/data2/hjs/pythonProject/pythonProject/Bird2026/input/train_5folds_sgkf.csv" # 分过 5 折的 csv
    train_audio_path = "/data2/hjs/pythonProject/pythonProject/Bird2026/input/train_audio"

    # train config
    model_name = "convnext_atto.d2_in1k"
    '''
    tf_efficientnet_b0.ns_jft_in1k
    convnext_atto.d2_in1k
    '''
    is_pretrained = True
    train_batch_size = 64
    valid_batch_size = 128
    now_cv = -np.inf
    epochs = 10
    start_lr_backbone = 1e-5
    start_lr_head = 1e-3
    min_lr_backbone = 1e-8
    min_lr_head = 1e-6
    scheduler = 'CosineAnnealingWithWarmupLR'
    n_accumulate = 1.0
    ckpt_save_path = None
    head_out = 234
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DataParallel = False

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
train = pd.read_csv(CONFIG.train_csv)
train['file_path'] = train['file_path'].str.replace('/kaggle/input/competitions/birdclef-2026', '/data2/hjs/pythonProject/pythonProject/Bird2026/input') # 将路径替换为本地路径

#########################################
# Dataset and DataLoader
#########################################
from zhiyue_dataset import Bird2026Dataset_1_1_4 as Bird2026Dataset

def prepare_loaders(df, fold=0):
    df_train = df[df["fold"] != fold]
    df_valid = df[df["fold"] == fold]
    
    train_datasets = Bird2026Dataset(df=df_train)
    valid_datasets = Bird2026Dataset(df=df_valid)
    
    train_loader = DataLoader(train_datasets, batch_size=CONFIG.train_batch_size, num_workers=CONFIG.n_workers, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_datasets, batch_size=CONFIG.valid_batch_size, num_workers=CONFIG.n_workers, shuffle=False, pin_memory=True)
    
    
    return train_loader, valid_loader

# 以下代码可检查Dataset，DataLoader是否实现基本功能
train_loader, valid_loader = prepare_loaders(train, 0)
x_train, y_train = next(iter(train_loader))
x_valid, y_valid = next(iter(valid_loader))
print(f"X_train shape : {x_train.shape}") # (batch_size, channels, H, W)
print(f"y_train shape : {y_train.shape}")
print(f"x_valid shape : {x_valid.shape}")
print(f"y_valid shape : {y_valid.shape}")

# 删除变量，回收垃圾
del train_loader, valid_loader, x_train, y_train, x_valid, y_valid
gc.collect()

#########################################
# Model
#########################################
class Bird2026Model(nn.Module):
    def __init__(self, model_name=CONFIG.model_name, is_pretrained=CONFIG.is_pretrained):
        super(Bird2026Model, self).__init__()
        self.backbone = timm.create_model(model_name=model_name, 
                                          pretrained=False)
        if is_pretrained: # 加载预训练权重
            state_dict = torch.load(f"/data2/hjs/pythonProject/pythonProject/Bird2026/ckpt/{model_name}.pth")
            self.backbone.load_state_dict(state_dict)

        if "efficientnet" in model_name:
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif "convnext" in model_name:
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
# Evaluation
#########################################
def CV_Score(y_trues: np.ndarray, y_preds: np.ndarray) -> float:
    """
    计算 BirdCLEF 2026 的官方评估指标：
    跳过没有真实正样本的类别，计算剩余类别的 Macro-averaged ROC-AUC。
    
    参数:
    y_trues: np.ndarray, 形状为 [n_samples, 234], 真实标签 (0 或 1)
    y_preds: np.ndarray, 形状为 [n_samples, 234], 模型预测概率
    
    返回:
    float: 最终的 Macro ROC-AUC 分数
    """
    # 1. 沿样本维度 (axis=0) 对真实标签求和，得到每个类别的正样本总数
    solution_sums = np.sum(y_trues, axis=0)
    
    # 2. 生成一个布尔掩码 (Boolean Mask)，标记出总数大于 0 的类别
    valid_classes_mask = solution_sums > 0
    
    # 对应官方代码的 assert len(scored_columns) > 0
    if not np.any(valid_classes_mask):
        raise ValueError("当前批次或验证集中没有任何正样本，无法计算 ROC-AUC。")
        
    # 3. 使用掩码过滤掉没有正样本的列
    y_trues_filtered = y_trues[:, valid_classes_mask]
    y_preds_filtered = y_preds[:, valid_classes_mask]
    
    # 4. 调用 sklearn 计算 Macro-averaged ROC-AUC
    score = roc_auc_score(y_trues_filtered, y_preds_filtered, average='macro')
    
    return float(score)

# 测试
y_trues = np.random.randint(0, 2, size=(40, 234))
y_preds = np.random.rand(40, 234)
score = CV_Score(y_trues, y_preds)
print(f"测试 CV_Score 函数，得到的分数为: {score:.4f}")

#########################################
# Train and Valid Function
#########################################
criterion = nn.BCEWithLogitsLoss(reduction='mean')

def train_one_epoch(model, optimizer, train_loader, epoch):
    model.train()
    
    y_preds = []
    y_trues = []
    
    dataset_size = 0
    running_loss = 0.0
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (images, labels) in bar:
        optimizer.zero_grad()
        
        batch_size = images.size(0)
        if CONFIG.DataParallel:
            images = images.cuda().float()
            labels = labels.cuda().float()
        else:
            images = images.to(CONFIG.device, dtype=torch.float)
            labels = labels.to(CONFIG.device, dtype=torch.float)
            
        outputs = model(images)
        loss = criterion(outputs, labels) / CONFIG.n_accumulate
        loss.backward()
        
        if (step + 1) % CONFIG.n_accumulate == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

        outputs = F.sigmoid(outputs)
        y_preds.append(outputs.detach().cpu().numpy())
        y_trues.append(labels.detach().cpu().numpy())

        if step < 5:
            train_cv = 0.0
        else:
            train_cv = CV_Score(np.concatenate(y_trues), np.concatenate(y_preds))

        running_loss += (loss.item() * batch_size)

        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch,
                        Train_Loss=epoch_loss,
                        Train_CV_AUC=train_cv,
                        LR_backbone=optimizer.optimizer1.param_groups[0]['lr'],
                        LR_head=optimizer.optimizer2.param_groups[0]['lr'])
    # Ensure that a parameter update is performed after the last accumulation cycle
    if (step + 1) % CONFIG.n_accumulate != 0:
        optimizer.step()
        optimizer.zero_grad()
        
    return epoch_loss, train_cv

def valid_one_epoch(model, optimizer, valid_loader, epoch):
    model.eval()
    
    y_preds = []
    y_trues = []
    dataset_size = 0
    running_loss = 0.0
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    with torch.no_grad():
        for step, (images, labels) in bar:
            batch_size = images.size(0)
            if CONFIG.DataParallel:
                images = images.cuda().float()
                labels = labels.cuda().float()
            else:
                images = images.to(CONFIG.device, dtype=torch.float)
                labels = labels.to(CONFIG.device, dtype=torch.float)

            outputs = model(images)
            loss = criterion(outputs, labels) / CONFIG.n_accumulate

            outputs = F.sigmoid(outputs)
            y_preds.append(outputs.detach().cpu().numpy())
            y_trues.append(labels.detach().cpu().numpy())
            if step < 5:
                valid_cv = 0.0
            else:
                valid_cv = CV_Score(np.concatenate(y_trues), np.concatenate(y_preds))
        
            running_loss += (loss.item() * batch_size)

            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(Epoch=epoch,
                            Valid_Loss=epoch_loss,
                            Valid_CV_AUC=valid_cv,
                            LR_backbone=optimizer.optimizer1.param_groups[0]['lr'],
                            LR_head=optimizer.optimizer2.param_groups[0]['lr'])
        

        y_preds = np.concatenate(y_preds)
        y_trues = np.concatenate(y_trues)
        cv = CV_Score(y_trues, y_preds) 
    
    return epoch_loss, cv

def get_time_fold():
    # Get the current time stamp
    current_time = time.time()
    print("Current timestamp:", current_time)
    
    # Convert a timestamp to a local time structure
    local_time = time.localtime(current_time)
    
    # Formatting local time
    CONFIG.formatted_time = time.strftime('%Y-%m-%d_%H:%M:%S', local_time)
    print("now time:", CONFIG.formatted_time)
    
    CONFIG.ckpt_save_path = f"output/{CONFIG.formatted_time}_{CONFIG.model_name}_output"
    if os.path.exists(CONFIG.ckpt_save_path) is False:
        os.makedirs(CONFIG.ckpt_save_path)

def run_training(fold, model, optimizer, train_loader, valid_loader, num_epochs=CONFIG.epochs, now_cv=CONFIG.now_cv):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {} x {}\n".format(torch.cuda.get_device_name(), torch.cuda.device_count()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_cv = now_cv
    best_model_path = None
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss, train_epoch_cv = train_one_epoch(model, optimizer, train_loader, epoch)
        valid_epoch_loss, valid_epoch_cv = valid_one_epoch(model, optimizer, valid_loader, epoch)
        print(f"epoch: {epoch}, LOSS = {valid_epoch_loss}, CV = {valid_epoch_cv}")
        
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(valid_epoch_loss)
        history['Train CV'].append(train_epoch_cv)
        history['Valid CV'].append(valid_epoch_cv)
        history['lr_backbone'].append(optimizer.optimizer1.param_groups[0]['lr'])
        history['lr_head'].append(optimizer.optimizer2.param_groups[0]['lr'])
        
        # deep copy the model
        if valid_epoch_cv >= best_epoch_cv:
            print(f"{b_}epoch: {epoch}, Validation CV Improved ({best_epoch_cv} ---> {valid_epoch_cv}))")
            best_epoch_cv = valid_epoch_cv
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "./{}/{}_CV_{:.4f}_Loss{:.4f}_epoch{:.0f}.pth".format(CONFIG.ckpt_save_path, fold, best_epoch_cv, valid_epoch_loss, epoch)
            if best_model_path is not None and os.path.exists(best_model_path): # 如果之前已经保存过 best_model，先删除旧的，节省磁盘空间
                os.remove(best_model_path) # 删除旧权重
            best_model_path = PATH
            torch.save(model.state_dict(), PATH)
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best CV: {:.4f}".format(best_epoch_cv))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history, best_model_path

#########################################
# Optimizer
#########################################
class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_epochs=10, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = T_max - warmup_epochs
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [(base_lr * (self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_epoch = self.last_epoch - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * cosine_epoch / self.cosine_epochs)) / 2 for base_lr in self.base_lrs]
        
# lr scheduler
def fetch_scheduler(optimizer, T_max, min_lr):
    if CONFIG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max, 
                                                   eta_min=min_lr)
    elif CONFIG.scheduler == "CosineAnnealingWithWarmupLR":
        scheduler = CosineAnnealingWithWarmupLR(optimizer, T_max=T_max, eta_min=min_lr, warmup_epochs=T_max//CONFIG.epochs)
        
    elif CONFIG.scheduler == None:
        return None
        
    return scheduler

class merge_optim():
    def __init__(self, optimizer1, optimizer2, lr_scheduler1=None, lr_scheduler2=None):
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.lr_scheduler1 = lr_scheduler1
        self.lr_scheduler2 = lr_scheduler2

    def zero_grad(self):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

    def step(self):
        self.optimizer1.step()
        self.optimizer2.step()
        if self.lr_scheduler1 is not None:
            self.lr_scheduler1.step()
        if self.lr_scheduler2 is not None:
            self.lr_scheduler2.step()

def get_optimizer(model, data_loader):
    if CONFIG.DataParallel:
        optimizer_backbone = optim.AdamW(model.module.backbone.parameters(), lr=CONFIG.start_lr_backbone)
        optimizer_head = optim.AdamW(model.module.head.parameters(), lr=CONFIG.start_lr_head)
    else:
        optimizer_backbone = optim.AdamW(model.backbone.parameters(), lr=CONFIG.start_lr_backbone)
        optimizer_head = optim.AdamW(model.head.parameters(), lr=CONFIG.start_lr_head)

    T_max = len(data_loader) * CONFIG.epochs
    scheduler_backbone = fetch_scheduler(optimizer_backbone, T_max=T_max, min_lr=CONFIG.min_lr_backbone)
    scheduler_head = fetch_scheduler(optimizer_head, T_max=T_max, min_lr=CONFIG.min_lr_head)
    
    optimizer = merge_optim(optimizer_backbone, optimizer_head, scheduler_backbone, scheduler_head)
    return optimizer

#########################################
# Start Training
#########################################
oof = []
true = []
historys = []
get_time_fold()
# 将当前.py文件备份到ckpt_save_path目录下
current_script_path = os.path.abspath(__file__)
output_txt_path = f'{CONFIG.ckpt_save_path}/current_code_backup.txt'
shutil.copy(current_script_path, output_txt_path)

for fold in range(0, CONFIG.n_folds): # CONFIG.n_folds
    print(f"==================== Train on Fold {fold+1} ====================")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = Bird2026Model()
    if CONFIG.DataParallel:
        device_ids = [0, 1]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda()
    else:
        model = model.to(CONFIG.device)
        
    train_loader, valid_loader = prepare_loaders(train, fold)
    optimizer = get_optimizer(model, train_loader)
    model, history, best_model_path = run_training(fold+1, model, optimizer, 
                                                   train_loader, valid_loader, 
                                                   num_epochs=CONFIG.epochs, now_cv=CONFIG.now_cv)
    historys.append(history)
    
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    with torch.no_grad():
        for step, (images, labels) in bar:
            batch_size = images.size(0)
            if CONFIG.DataParallel:
                images = images.cuda().float()
                labels = labels.cuda().float()
            else:
                images = images.to(CONFIG.device, dtype=torch.float)
                labels = labels.to(CONFIG.device, dtype=torch.float)

            outputs = model(images)
            outputs = F.sigmoid(outputs)
            oof.append(outputs.detach().cpu().numpy())
            true.append(labels.detach().cpu().numpy())
        print()

oof = np.concatenate(oof)
true = np.concatenate(true)

#########################################
# Local CV
#########################################
local_cv = CV_Score(true, oof)
print("Local CV : ", local_cv)

np.save(f"{CONFIG.ckpt_save_path}/true.npy", true)
np.save(f"{CONFIG.ckpt_save_path}/oof.npy", oof)
