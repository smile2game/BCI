'''
Coding: utf-8
Author: vector-wlc
Date: 2022-12-30 20:28:52
Description: 
'''
import torch
import numpy as np


def load_data(npy_dir: str, train_rate: float = 0.75, batch_size: int = 64) -> tuple:
    x = np.load(npy_dir + "x.npy")
    y = np.load(npy_dir + "y.npy")
    # 由于 x 的数据维度要求为 N 1 C T， 至于为什么第二维多了一个看似没有意义的 1
    # 是因为在图像处理那边，图片的数据集的维度就是 N 3 W H，3 就是图片的 RGB 通道

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y)
    x = torch.unsqueeze(x, 1)

    total_num = len(x)
    train_num = int(total_num * train_rate)
    train_dataset = torch.utils.data.TensorDataset(
        x[:train_num], y[:train_num])
    test_dataset = torch.utils.data.TensorDataset(
        x[train_num:], y[train_num:])
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_data_loader, test_data_loader


def train_model(train_loader, model, epochs):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters()
    )

    for epoch in range(epochs):
        running_loss = 0.0
        batch_size = None
        size_loss = 0
        for index, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            model_out = model(x)

            loss = loss_func(model_out, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
        print(
            f'\repochs: {epoch + 1} / {epochs}\tTrain loss: {running_loss:.4f}', end='')
    print('\nFinish Training!')


def test_model(test_loader, model):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    correct_num = 0
    total_num = 0
    for index, data in enumerate(test_loader):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        model_out = model(x)

        _, pred = torch.max(model_out, 1)
        correct_num += np.sum(
            pred.cpu().numpy() == y.cpu().numpy()
        )
        total_num += len(y)
    print('\nTest acc: ' + str(correct_num / total_num))
