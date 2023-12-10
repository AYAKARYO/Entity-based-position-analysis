import json
import random

with open("test_data\\all.json", encoding="utf-8") as f:
    file = json.load(f)
    random.shuffle(file)
    print(file.__len__())
    train = open("test_data\\train.json", encoding="utf-8", mode="w+")
    dev = open("test_data\\dev.json", encoding="utf-8", mode="w+")
    test = open("test_data\\test.json", encoding="utf-8", mode="w+")
    train_list = []
    dev_list = []
    test_list = []
    train_num = int(file.__len__() * 0.7)
    dev_num = int(file.__len__() * 0.9)
    for i in range(train_num):
        train_list.append(file[i])
    for i in range(train_num, dev_num):
        dev_list.append(file[i])
    for i in range(dev_num, file.__len__()):
        test_list.append(file[i])
    json.dump(train_list, train)
    json.dump(dev_list, dev)
    json.dump(test_list, test)
