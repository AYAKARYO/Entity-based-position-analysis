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
    model.train()
    for epoch in range(epochs):
        loss_sum = 0
        for batch in train_iter:
            input_, entity, events, label = batch
            optimizer.zero_grad()
            output = model(input_, entity, events)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print("epoch: ", epoch, "loss:", loss_sum / len(test_iter))
        writer.add_scalar("train_loss", loss.item(), epoch)
        if (epoch + 1) % int(config["model"]["test_internal"]) == 0:
            test_acc, recall, f1, cm = evaluate(epoch)
            writer.add_scalar("test_acc", test_acc, epoch + 1)
            writer.add_scalar("recal_rates", recall, epoch + 1)
            writer.add_scalar("f1_score", f1, epoch + 1)
            cm_array = np.array(cm)
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
            writer.add_figure("Confusion Matrix", plt.gcf(), epoch + 1)
            print("test_acc:", test_acc)


def evaluate(epoch):
    model.eval()
    total_acc, total_count = 0, 0
    loss_sum = 0
    with torch.no_grad():
        predicted_labels = []
        predicted = []
        true_labels = []
        for batch in test_iter:
            input_, entity, aspect, label = batch
            true_labels.append(label)
            predicted_label = model(input_, entity, aspect)
            predicted_labels.append(predicted_label)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            loss_sum += loss.item()
        print(str(epoch + 1) + ":")
        print("test_total_count:", total_count)
        print("test_loss:", loss_sum / len(test_iter))
        true_labels_flat = [
            item
            for sublist in true_labels
            for tensor in sublist
            for item in tensor.view(-1).cpu().numpy()
        ]
        print("true_labels_flat:", true_labels_flat)
        predicted_labels_flat = [
            item
            for sublist in predicted_labels
            for tensor in sublist
            for item in tensor.view(-1).cpu().numpy()
        ]
        print("predicted_labels_flat:", predicted_labels_flat)
        arr = np.array(predicted_labels_flat)
        max_indices = []
        for i in range(0, len(arr), 3):
            max_index = np.argmax(arr[i : i + 3])
            max_indices.append(max_index)
        print("pre_max_indices:", max_indices)
        list_len = max_indices.__len__()
        max_indices = max_indices[:list_len]
        f1 = f1_score(
            true_labels_flat, max_indices[:list_len], average="macro", zero_division=1
        )
        recall = recall_score(
            true_labels_flat, max_indices[:list_len], average="macro", zero_division=1
        )
        cm = confusion_matrix(true_labels_flat, max_indices)
        indices = range(len(true_labels_flat))
        fig, ax = plt.subplots()
        ax.plot(indices, true_labels_flat, label="True Labels")
        ax.plot(indices, max_indices, label="Max Indices")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.legend()
        fig.set_size_inches(15, 3)
        writer.add_figure("Arrays Comparison", fig, epoch + 1)
        for i in range(len(true_labels_flat)):
            writer.add_scalars(
                "Arrays Comparison",
                {"True Labels": true_labels_flat[i], "Max Indices": max_indices[i]},
                i + 1,
            )
        cnt = 0
        for i in range(list_len):
            if true_labels_flat[i] == max_indices[i]:
                cnt += 1
        print(f"召回率: {recall}")
        print(f"F1分数: {f1}")
        print(f"混淆矩阵: \n{cm}")
        print("cnt/287:", cnt / 287)
    model.train()
    return total_acc / total_count, recall, f1, cm


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    seed = int(config["model"]["seed"])
    TORCH_SEED = seed
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["PYTHONHASHSEED"] = str(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_iter = pickle.load(open(config["data"]["train_loader"], mode="rb"))
    test_iter = pickle.load(open(config["data"]["test_loader"], mode="rb"))
    embbedding = pickle.load(open(config["data"]["emb_path"], mode="rb"))
    input_size = int(config["model"]["input_size"])
    hidden_size = int(config["model"]["hidden_size"])
    model = ATAELSTM_Network(
        emb_matrix=embbedding, input_size=input_size, hidden_size=hidden_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = float(config["model"]["lr"])
    weight_decay = float(config["model"]["weight_decay"])
    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    writer = SummaryWriter(
        config["data"]["log_path"]
        + str(lr)
        + " "
        + str(input_size)
        + " "
        + str(hidden_size)
    )
    writer.add_scalar("seed", seed, 0)
    writer.add_scalar("learning_rate", lr, 0)
    writer.add_scalar("weight_decay:", weight_decay, 0)
    epochs = int(config["model"]["epochs"])
    train(epochs)
    writer.close()
