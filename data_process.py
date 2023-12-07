from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import linecache
import json
import numpy as np
import torch
import pickle
from gensim.models import Word2Vec
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # 查看我们使用的设备，如果是cpu证明GPU没有开启，请自行检查原因
PAD, UNK = "<pad>", "<unk>"  # 定义特殊token
# vocab = {PAD: 0, UNK: 1}
vocab = pickle.load(open(config["data"]["vocabulary_path"], "rb"))  # 加载字典


# 定义自己的Dataset类，用于加载和处理数据
class MyDataset(Dataset):
    def __init__(self, path):
        self.tokenizer = get_tokenizer("basic_english")  # 分词器
        data = []
        # 读取数据并且处理数据
        with open(path, "r", encoding="utf-8") as f:
            self.file = json.load(f)
            for line in self.file:
                attitude = 0
                if line["sentiment"] == "NEG":
                    attitude = 0
                elif line["sentiment"] == "POV":
                    attitude = 1
                elif line["sentiment"] == "NOR":
                    attitude = 2
                data.append(
                    (
                        self.tokenizer(line["text_content"]),
                        self.tokenizer(line["entity"]),
                        self.tokenizer(line["events"]),
                        attitude,
                    )
                )
        self.text_list, self.entity_list, self.events_list, self.label_list = zip(*data)
        self.vocab = vocab
        self.text_list_ = []
        self.entity_list_ = []
        self.events_list_ = []
        for i in range(len(self.text_list)):
            self.text_list_.append(
                torch.tensor(
                    [
                        vocab[token] if token in list(vocab.keys()) else vocab[UNK]
                        for token in self.text_list[i]
                    ]
                )
            )
            self.entity_list_.append(
                torch.tensor(
                    [
                        vocab[token] if token in list(vocab.keys()) else vocab[UNK]
                        for token in self.entity_list[i]
                    ]
                )
            )
            self.events_list_.append(
                torch.tensor(
                    [
                        vocab[token] if token in list(vocab.keys()) else vocab[UNK]
                        for token in self.events_list[i]
                    ]
                )
            )  # 使用pad_sequence函数将这个列表中的所有序列填充到相同的长度
        self.text_list_ = pad_sequence(
            self.text_list_, batch_first=True, padding_value=0
        )

        # 将aspect_list和label_list转换为PyTorch张量

        self.entity_list_ = torch.tensor(self.entity_list_, dtype=torch.long)
        self.events_list_ = torch.tensor(self.events_list_, dtype=torch.long)
        self.label_list = torch.tensor(self.label_list, dtype=torch.long)
        self.text_list_ = [text.to(device) for text in self.text_list_]
        self.entity_list_ = self.entity_list_.to(device)
        self.events_list_ = self.events_list_.to(device)
        self.label_list = self.label_list.to(device)

    # 获取数据长度
    def __len__(self):
        return len(self.file)

    # 按照index获取数据
    def __getitem__(self, idx):
        text = self.text_list_[idx]
        entity = self.entity_list_[idx]
        events = self.events_list_[idx]
        label = self.label_list[idx]
        return text, entity, events, label


# 用于DataLoader装载数据时进一步处理batch数据
def batch_process(batch):
    text_list, aspect_list, label_list = zip(*batch)
    text_list_ = []
    aspect_list_ = []

    # token转化成id
    for i in range(len(text_list)):
        text_list_.append(
            torch.tensor(
                [
                    vocab[token] if token in list(vocab.keys()) else vocab[UNK]
                    for token in text_list[i]
                ]
            )
        )
        aspect_list_.append(
            torch.tensor(
                [
                    vocab[token] if token in list(vocab.keys()) else vocab[UNK]
                    for token in aspect_list[i]
                ]
            )
        )

    text_list_ = pad_sequence(
        text_list_, batch_first=True, padding_value=0
    )  # padding数据

    # 将数据类型转化成tensor
    aspect_list_ = torch.tensor(aspect_list_, dtype=torch.long)
    label_list = torch.tensor(label_list, dtype=torch.long)

    return text_list_.to(device), aspect_list_.to(device), label_list.to(device)
