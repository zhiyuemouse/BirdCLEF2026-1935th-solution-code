import numpy as np
from sklearn.metrics import roc_auc_score

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

cal_dir =  "/data2/hjs/pythonProject/pythonProject/Bird2026/output/2026-04-05_23:53:41_convnext_atto.d2_in1k_output"
y_trues = np.load(f"{cal_dir}/true.npy")  # 形状 [n_samples, 234]
y_preds = np.load(f"{cal_dir}/oof.npy")  # 形状 [n_samples, 234]

cv_score = CV_Score(y_trues, y_preds)
print(f"CV Score: {cv_score:.4f}")
