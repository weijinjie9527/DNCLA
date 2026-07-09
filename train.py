import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import numpy as np
import csv
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import random_split
from Datasets.DataReader import Datasets
from models.CLA_TFBS import CLAModel


class Constructor:
    def __init__(self, model_class, dataset_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class().to(self.device)
        self.dataset_name = dataset_name
        self.batch_size = 64
        self.epochs = 10
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.loss_function = nn.BCELoss()

    def calculate_metrics(self, labels, preds):
        acc = accuracy_score(labels, np.round(preds))
        roc = roc_auc_score(labels, preds)
        precision, recall, _ = precision_recall_curve(labels, preds)
        pr_auc = auc(recall, precision)
        return acc, roc, pr_auc

    def practise(self, TrainLoader, ValidateLoader):
        best_val_auc = 0
        os.makedirs('save_models', exist_ok=True)
        save_path = os.path.join('save_models', f'best_{self.dataset_name}.pth')

        for epoch in range(self.epochs):
            self.model.train()
            process_bar = tqdm(TrainLoader, unit='batch', mininterval=1.0, leave=False)
            process_bar.set_description(f"Epoch [{epoch + 1}/{self.epochs}]")

            train_loss = 0
            for ncp, dpcp, label in process_bar:
                ncp, dpcp, label = ncp.to(self.device), dpcp.to(self.device), label.to(self.device).float()

                self.optimizer.zero_grad()
                # 确保顺序：ncp (9通道), dpcp (15通道)
                output = self.model(ncp, dpcp).squeeze(-1)
                loss = self.loss_function(output, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                train_loss += loss.item()
                # 在进度条右侧实时更新 Loss
                process_bar.set_postfix(loss=f"{loss.item():.4f}")

            self.scheduler.step()

            # 验证阶段
            self.model.eval()
            val_p, val_l = [], []
            with torch.no_grad():
                for vn, vd, vl in ValidateLoader:
                    vo = self.model(vn.to(self.device), vd.to(self.device)).squeeze(-1)
                    val_p.extend(vo.cpu().numpy())
                    val_l.extend(vl.numpy())

            acc, roc, pr = self.calculate_metrics(val_l, val_p)

            # 每个 Epoch 结束后打印完整指标
            print(f"  -> Validation Metrics: ACC={acc:.4f}, ROC={roc:.4f}, PR={pr:.4f}")

            if roc > best_val_auc:
                best_val_auc = roc
                torch.save(self.model.state_dict(), save_path)

    def test(self):
        save_path = os.path.join('save_models', f'best_{self.dataset_name}.pth')
        self.model.load_state_dict(torch.load(save_path))
        self.model.eval()

        test_loader = loader.DataLoader(Datasets(self.dataset_name, Test=True), batch_size=32)
        tp, tl = [], []
        with torch.no_grad():
            for tn, td, tlb in test_loader:
                to = self.model(tn.to(self.device), td.to(self.device)).squeeze(-1)
                tp.extend(to.cpu().numpy())
                tl.extend(tlb.numpy())
        return self.calculate_metrics(tl, tp)

if __name__ == '__main__':
    data_list_path = 'FILESLIST.txt'
    with open(data_list_path, 'r') as f:
        datasets = [line.strip() for line in f if line.strip()]

    all_results = []
    # 准备 CSV 文件表头
    csv_file = 'evaluation_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'ACC', 'ROC_AUC', 'PR_AUC'])

    for i, name in enumerate(datasets):
        print(f"\n[{i + 1}/{len(datasets)}] Processing: {name}")

        try:
            trainer = Constructor(CLAModel, name)

            # 数据准备
            full_db = Datasets(name, Test=False)
            t_size = int(len(full_db) * 0.8)
            v_size = len(full_db) - t_size
            t_db, v_db = random_split(full_db, [t_size, v_size])

            t_ld = loader.DataLoader(t_db, batch_size=64, shuffle=True)
            v_ld = loader.DataLoader(v_db, batch_size=64)

            # 训练与测试
            trainer.practise(t_ld, v_ld)
            acc, roc, pr = trainer.test()

            # 1. 存入内存列表用于最后计算平均值
            all_results.append([name, acc, roc, pr])

            # 2. 实时写入 CSV，防止崩溃丢失数据
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, f"{acc:.4f}", f"{roc:.4f}", f"{pr:.4f}"])

            print(f"Done! {name} -> ACC: {acc:.4f} | ROC: {roc:.4f} | PR: {pr:.4f}")

            # 释放显存（可选，预防OOM）
            del trainer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

    # --- 所有数据集跑完后的汇总展示 ---
    print("\n" + "=" * 50)
    print(f"{'Dataset Name':<20} | {'ACC':<8} | {'ROC':<8} | {'PR':<8}")
    print("-" * 50)

    for res in all_results:
        print(f"{res[0]:<20} | {res[1]:<8.4f} | {res[2]:<8.4f} | {res[3]:<8.4f}")

    print("-" * 50)

    # 计算并打印平均值
    if all_results:
        avg_metrics = np.mean([res[1:] for res in all_results], axis=0)
        print(f"AVERAGE (Total {len(all_results)} datasets):")
        print(f"ACC: {avg_metrics[0]:.4f} | ROC: {avg_metrics[1]:.4f} | PR: {avg_metrics[2]:.4f}")
    print("=" * 50)
