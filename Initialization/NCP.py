import numpy as np


def NCP(seq):
    # 1. 标准 NCP 映射表
    prop_map = {
        'A': [1, 0, 1], 'C': [0, 1, 1],
        'G': [1, 1, 0], 'T': [0, 0, 0]
    }

    # 2. 准备滑动窗口存储特征
    # 101bp -> 99个 3-mer
    seq_len = len(seq)
    window_size = 3
    new_len = seq_len - window_size + 1

    # 初始化 9x99 的矩阵
    feat = np.zeros((9, new_len))

    for i in range(new_len):
        # 提取当前 3-mer
        sub_seq = seq[i:i + window_size]

        # 获取三个碱基的 NCP 特征并拼接成 9 维向量
        # 例如: [1,0,1] + [0,1,1] + [1,1,0] -> [1,0,1,0,1,1,1,1,0]
        tri_feat = []
        for base in sub_seq:
            tri_feat.extend(prop_map.get(base, [0, 0, 0]))

        feat[:, i] = tri_feat

    return feat  # 返回形状 (9, 99)