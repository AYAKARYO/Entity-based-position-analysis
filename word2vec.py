from nltk.tokenize import word_tokenize
import json
import pickle
import nltk
from gensim.models import word2vec
import numpy as np
import configparser
import random

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    seed = int(config["model"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    config = configparser.ConfigParser()
    config.read("config.ini")
    nltk.download("punkt")
    PAD, UNK = "<pad>", "<unk>"  # 定义特殊token
    with open(config["data"]["all_path"], encoding="utf-8", mode="r") as f:
        file = json.load(f)
        sentences_list = []
        for item in file:
            sentences_list.append(item["text_content"])
        sentences_list.append("Sino-US-relations")
        sentences_list.append("relations-across-the-Taiwan-Straits")
        sentences_list.append("vaccine")
        tokenized_word_list = [word_tokenize(doc) for doc in sentences_list]
        vocab = {}
        vocab[PAD] = 0
        vocab[UNK] = 1
        for line in tokenized_word_list:
            for word in line:
                if word not in vocab:
                    vocab[word] = vocab.__len__()
        model = word2vec.Word2Vec(
            sentences=vocab,
            size=int(config["model"]["input_size"]),
            window=int(config["model"]["windows"]),
            min_count=int(config["model"]["min_count"]),
        )
        model.train(
            sentences=vocab, total_examples=model.corpus_count, epochs=model.epochs
        )
        emb_matrix = np.zeros((len(vocab), int(config["model"]["input_size"])))
        for i, word in enumerate(model.wv.vocab):
            emb_matrix[i] = model.wv[word]
        pickle.dump(emb_matrix, open(config["data"]["emb_path"], mode="wb"))
        pickle.dump(vocab, open(config["data"]["vocabulary_path"], "wb"))
