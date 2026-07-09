import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
# 确保以下包路径正确
from wjjtfbs.Initialization.NCP import NCP
from wjjtfbs.Initialization.DPCP import DPCP

class SampleReader:
    def __init__(self, folder_name):
        # 对应路径：项目根目录/Datasets/data/Sequence/[folder_name]/Sequence/
        # 使用 os.path.realpath(__file__) 确保从 DataReader.py 所在位置开始查找
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.seq_path = os.path.join(current_dir, 'data', 'Sequence', folder_name, 'Sequence')

    def get_features(self, Test=False):
        csv_name = 'Test_seq.csv' if Test else 'Train_seq.csv'
        file_full_path = os.path.join(self.seq_path, csv_name)

        if not os.path.exists(file_full_path):
            raise FileNotFoundError(f"找不到文件: {file_full_path}")

        row_data = pd.read_csv(file_full_path, sep=' ', header=None)
        ncp_list, dpcp_list, label_list = [], [], []

        for i in range(len(row_data)):
            sequence = str(row_data.loc[i, 1]).upper()
            label = int(row_data.loc[i, 2])
            ncp_list.append(NCP(sequence))
            dpcp_list.append(DPCP(sequence))
            label_list.append(label)

        return (np.array(ncp_list, dtype=np.float32),
                np.array(dpcp_list, dtype=np.float32),
                np.array(label_list, dtype=np.float32))

class Datasets(Dataset):
    def __init__(self, folder_name, Test=False):
        reader = SampleReader(folder_name)
        self.ncp, self.dpcp, self.labels = reader.get_features(Test=Test)

    def __getitem__(self, index):
        return self.ncp[index], self.dpcp[index], self.labels[index]

    def __len__(self):
        return len(self.labels)