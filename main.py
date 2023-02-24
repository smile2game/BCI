'''
Coding: utf-8
Author: vector-wlc
Date: 2022-12-30 21:02:35
Description: 
'''
from tools import *
from model import DeepConvNet


if __name__ == "__main__":
    subject_id = 1
    train_data_loader, test_data_loader = load_data(
        f"data/preprocess/S{subject_id:>02d}/")
    model = DeepConvNet(
        sample=256, class_num=2,
        time_channels=25, time_size=(1, 9),
        spatial_channels=50, spatial_size=(64, 1),
        feature_pool_size=(1, 3), feature_channels_list=[100, 200], dropout=0.5)

    train_model(train_data_loader, model, 50)
    test_model(test_data_loader, model)
