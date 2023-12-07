from data_process import *
from model import *
import os
import pickle
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tensorboardX import SummaryWriter
import configparser
from gensim.models import Word2Vec

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# 训练函数
def train(epochs):
    model.train()  # 模型设置成训练模式
    for epoch in range(epochs):  # 训练epochs轮
        loss_sum = 0  # 记录每轮loss
        for batch in train_iter:
            input_, entity, events, label = batch
            optimizer.zero_grad()  # 每次迭代前设置grad为0
            # output = model(input_)
            output = model(input_, entity, events)
            loss = criterion(output, label)  # 计算loss
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            loss_sum += loss.item()  # 累积loss
        print("epoch: ", epoch, "loss:", loss_sum / len(test_iter))
        writer.add_scalar("train_loss", loss.item(), epoch)
        if (epoch + 1) % int(config["model"]["test_internal"]) == 0:
            test_acc, recall, f1, cm = evaluate(epoch)
            writer.add_scalar("test_acc", test_acc, epoch + 1)
            writer.add_scalar("recal_rates", recall, epoch + 1)
            writer.add_scalar("f1_score", f1, epoch + 1)
            # writer.add_scalar("confusion_matrix", cm, epoch + 1)
            # Convert confusion matrix to a NumPy array
            cm_array = np.array(cm)

            # Plot and save the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm_array,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["0", "1"],
                yticklabels=["0", "1"],
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")

            # Use add_figure to add the confusion matrix to TensorBoard
            writer.add_figure("Confusion Matrix", plt.gcf(), epoch + 1)

            print("test_acc:", test_acc)


# 测试函数
def evaluate(epoch):
    model.eval()
    total_acc, total_count = 0, 0
    loss_sum = 0
    with torch.no_grad():  # 测试时不计算梯度
        predicted_labels = []
        predicted = []
        true_labels = []
        for batch in test_iter:
            input_, entity, aspect, label = batch
            true_labels.append(label)
            predicted_label = model(input_, entity, aspect)
            predicted_labels.append(predicted_label)
            loss = criterion(predicted_label, label)  # 计算loss
            total_acc += (predicted_label.argmax(1) == label).sum().item()  # 累计正确预测数
            total_count += label.size(0)  # 累积总数
            loss_sum += loss.item()  # 累积loss
        print(str(epoch + 1) + ":")
        print("test_loss:", loss_sum / len(test_iter))
        predicted_labels = predicted_label.argmax(1).detach().cpu().numpy()
        true_labels = label.detach().cpu().numpy()
        f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=1)
        recall = recall_score(
            true_labels, predicted_labels, average="macro", zero_division=1
        )
        cm = confusion_matrix(true_labels, predicted_labels)

        print(f"召回率: {recall}")
        print(f"F1分数: {f1}")
        print(f"混淆矩阵: \n{cm}")
    model.train()
    return total_acc / total_count, recall, f1, cm


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
    # batch_size = int(config["data"]["batch_size"])
    # train_dataset = MyDataset(config["data"]["train_path"])
    # test_dataset = MyDataset(config["data"]["test_path"])

    # train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_iter = pickle.load(open(config["data"]["train_loader"], mode="rb"))
    test_iter = pickle.load(open(config["data"]["test_loader"], mode="rb"))
    # 加载我们的Embedding矩阵
    embbedding = pickle.load(open(config["data"]["emb_path"], mode="rb"))
    # embedding = torch.tensor(
    #    np.load(config["data"]["emb_path"])["embeddings"], dtype=torch.float
    # )
    # 定义模型
    input_size = int(config["model"]["input_size"])
    hidden_size = int(config["model"]["hidden_size"])
    # model = LSTM_Network(embedding, input_size, hidden_size).to(device)
    # model = AELSTM_Network(embedding, input_size, hidden_size).to(device)
    model = ATAELSTM_Network(
        emb_matrix=embbedding, input_size=input_size, hidden_size=hidden_size
    ).to(device)

    # 定义loss函数、优化器
    criterion = nn.CrossEntropyLoss()
    lr = float(config["model"]["lr"])
    weight_decay = float(config["model"]["weight_decay"])
    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    writer = SummaryWriter(
        config["data"]["log_path"] + str(lr) + " " + str(input_size) + " " + str(hidden_size)
    )
    writer.add_scalar("seed", seed, 0)
    writer.add_scalar("learning_rate", lr, 0)
    writer.add_scalar("weight_decay:", weight_decay, 0)
    epochs = int(config["model"]["epochs"])
    train(epochs)
    writer.close()
