# 评估指标模块：算 IoU / F1 / Precision / Recall（分割任务标准指标）
import numpy as np


def compute_metrics(pred, target, num_classes=2):
    """
    计算精确率、召回率、F1 Score、IoU
    pred:   模型预测的类别索引，形状任意（如 (128,128)），元素 0 或 1
    target: 真值类别索引，形状和 pred 一样，元素 0 或 1
    返回: 一个字典，包含每个类的四项指标 + 滑坡类主指标 + 平均 IoU
    """
    pred = np.asarray(pred).reshape(-1)    # 展平成一维数组，每个元素是一个像素的类别
    target = np.asarray(target).reshape(-1)

    # 创建真正例/假正例/假负例/真负例变量，第一个字母代表
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    tn = np.zeros(num_classes)

    for c in range(num_classes):
        # 逐类统计上面四个数（& 是按位与，表示"同时满足两个条件"）
        tp[c] = np.sum((pred == c) & (target == c))    # TP：预测是c且真值也是c（预测对了）
        fp[c] = np.sum((pred == c) & (target != c))    # FP：预测是c但真值不是c（把别的错认成 c）
        fn[c] = np.sum((pred != c) & (target == c))    # FN：真值是c但预测不是 c（漏掉了 c）
        tn[c] = np.sum((pred != c) & (target != c))    # TN：真值不是c且预测也不是c（正确排除）

    results = {}  # 用来装最终结果的字典
    for c in range(num_classes):
        # 1e-14 是防止除以 0 的小量
        p = tp[c] / (tp[c] + fp[c] + 1e-14)    # 精确率 Precision = TP/(TP+FP)
        r = tp[c] / (tp[c] + fn[c] + 1e-14)    # 召回率 Recall   = TP/(TP+FN)
        f1 = 2 * p * r / (p + r + 1e-14)       # F1 = 2PR/(P+R)，综合 P 和 R
        iou = tp[c] / (tp[c] + fp[c] + fn[c] + 1e-14)  # IoU = TP/(TP+FP+FN)，交并比
        results[f"class_{c}"] = {"precision": p, "recall": r, "f1": f1, "iou": iou}

    # 滑坡是第 1 类（name_classes=['Non-Landslide','Landslide']）
    results["landslide_f1"] = results["class_1"]["f1"]
    results["landslide_iou"] = results["class_1"]["iou"]
    results["mean_iou"] = np.mean([results[f"class_{c}"]["iou"] for c in range(num_classes)])
    return results


# 自测：用随机数据验证指标能正常算出来
if __name__ == "__main__":
    pred = np.random.randint(0, 2, (128, 128))  # 随机预测的类别图
    target = np.random.randint(0, 2, (128, 128))  # 随机真值
    m = compute_metrics(pred, target, num_classes=2)
    print("滑坡类 F1 :", round(float(m["landslide_f1"]), 4))
    print("滑坡类 IoU:", round(float(m["landslide_iou"]), 4))