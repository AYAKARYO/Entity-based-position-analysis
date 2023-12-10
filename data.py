import configparser
import os
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
from data_process import MyDataset


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    seed = int(config["model"]["seed"])
    TORCH_SEED = seed
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置模型在几号GPU上跑
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置device

    # 设置随机数种子，保证结果一致
    os.environ["PYTHONHASHSEED"] = str(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 创建数据集
    test_dataset = MyDataset(config["data"]["test_path"])
    train_dataset = MyDataset(config["data"]["train_path"])
    dev_dataset = MyDataset(config["data"]["dev_path"])
    batch_size = int(config["data"]["batch_size"])
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    dev_iter = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    pickle.dump(train_iter, open(config["data"]["train_loader"], mode="wb"))
    pickle.dump(test_iter, open(config["data"]["test_loader"], mode="wb"))
    pickle.dump(dev_iter, open(config["data"]["dev_loader"], mode="wb"))
