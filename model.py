import torch
import torch.nn as nn
import pickle
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read("config.ini")

vocab = pickle.load(open(config["data"]["vocabulary_path"], "rb"))  # 加载字典


# LSTM模型
class ATAELSTM_Network(nn.Module):
    def __init__(self, emb_matrix, input_size, hidden_size):
        super(ATAELSTM_Network, self).__init__()
        self.embedding = nn.Embedding(vocab.__len__(), input_size)

        if emb_matrix is not None:
            emb_matrix = torch.tensor(emb_matrix, dtype=torch.float32)
            self.embedding.weight.data.copy_(emb_matrix)
        self.lstm = nn.GRU(
            input_size=input_size * 3,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=3)

        # 用于attention的参数
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()  # 激活函数
        self.softmax = nn.Softmax(dim=-1)  # softmax

        self.w = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.w.data.normal_(mean=0.0, std=0.02)

    def forward(self, input, entity, events):
        input = self.embedding(input)
        batch_size, seq_len, emb_ = input.size()
        events = self.embedding(events)
        events = events.unsqueeze(1).expand(-1, seq_len, -1)
        entity = self.embedding(entity)
        entity = entity.unsqueeze(1).expand(-1, seq_len, -1)

        input = torch.cat((input, entity, events), dim=-1)  # 将句子和aspect按最后一维度拼接
        output, _ = self.lstm(input)  # [batch, seq, embedding]

        output_2 = self.attention(output)  # attention机制
        output = torch.cat(
            (output_2, output[:, -1, :]), dim=-1
        )  # 论文中的操作，将attention后的表示和lstm最后的hidden进行拼接
        output = self.tanh(self.fc(output))

        return output

    def attention(self, K, mask=None):
        batch_size, seq_len, emb_size = K.size()

        K_ = K.reshape(-1, emb_size)
        K_ = self.linear(K_)
        K_ = self.tanh(K_)

        attention_matrix = torch.mm(K_, self.w).view(batch_size, -1)  # [batch, len]

        if mask is not None:
            mask = mask.bool()
            attention_matrix.masked_fill_(mask, -float("inf"))

        attention_matrix = self.softmax(attention_matrix)  # 生成attention矩阵
        output = torch.bmm(attention_matrix.unsqueeze(1), K)  # [batch, 1, emb_size]
        output = output.squeeze(1)  # [batch, emb_size]

        return output
