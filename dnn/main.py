#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/12/3 下午8:03 
"""

from utils import *
from model import MyDnn, MyDnnSimple, FocalLoss
from parser import parameter_parser

import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE

from tqdm import tqdm
import numpy as np
import time
import json


def train(args, time_span, version, category):
    if version == 0:
        device = 'cuda:0'
    else:
        device = 'cuda:1'
    print(device)
    # data_process process
    print('data_process process')
    x_list = np.load(f'data/x_{time_span}.npy')
    y_list = np.load(f'data/y_{time_span}.npy')
    x_list = torch.Tensor(x_list)
    y_list = torch.LongTensor(y_list)
    y_list = label_deal(y_list, category)

    sm = SMOTE(random_state=int(time_span),
               sampling_strategy={
                   0: int(torch.sum(y_list == 0)),
                   1: int(torch.sum(y_list == 1)) * 5}
               )
    x_list, y_list = sm.fit_resample(x_list, y_list)

    x_list = torch.Tensor(x_list)
    y_list = torch.LongTensor(y_list)

    print('  x shape', x_list.shape, 'y shape', y_list.shape)
    print('  pos/neg', torch.sum(y_list == 1), torch.sum(y_list == 0))
    data = MyDataSet(x_list, y_list)
    train_data, dev_data = random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])
    # category2vec
    c = np.load(f'data/category2vec_{time_span}.npy')
    c = torch.Tensor(c[category])
    c = c.to(device)

    # dataloader
    print('dataloader')
    print('  train data_process')
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    print('  dev data_process')
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size)

    # model
    print('model')
    model = MyDnnSimple()
    # model_state_dict = torch.load('../model/rope_model.pt')
    # model.load_state_dict(model_state_dict)
    model.to(device)

    # loss
    weights = torch.Tensor([1, torch.sum(y_list == 0) / torch.sum(y_list == 1)]).to(device)
    print(weights)
    fun_loss = nn.BCEWithLogitsLoss(pos_weight=weights[1])
    # fun_loss = nn.BCEWithLogitsLoss()

    # optimizer
    print('optimizer')
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                  factor=args.lr_reduce_factor,
    #                                                  patience=args.lr_schedule_patience)

    print('model initialized')
    print('model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # train
    print('training...')
    print('batch_size:', args.batch_size, 'epochs:', args.epochs)
    scaler = GradScaler()

    best_f1 = 0
    best_f1_list = []
    best_precision = 0
    bast_recall = 0

    loss_list = []

    for epoch in range(args.epochs):
        # train
        model.train()
        loss_collect = []
        with tqdm(total=len(train_dataloader), desc='train---epoch:{}'.format(epoch)) as bar:
            for step, (x, y) in enumerate(train_dataloader):
                x = x.to(device)
                y = y.to(device)
                # y for BCE
                y = y.unsqueeze(1).float()
                # y trans for loss
                # y = y.unsqueeze(1).float()
                with autocast():
                    outputs = model(x, c)
                    # loss = criterion(outputs, y)
                    loss = fun_loss(outputs, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # scheduler.step(loss.item())
                loss_collect.append(loss.item())
                bar.update(1)
                bar.set_postfix(loss=np.mean(loss_collect), lr=optimizer.param_groups[0]['lr'])
        loss_list.append(np.mean(loss_collect))

        # dev
        model.eval()
        f1_collect = []
        recall_collect = []
        precision_collect = []

        with tqdm(total=len(dev_dataloader), desc='dev') as bar:
            for step, (x, y) in enumerate(dev_dataloader):
                x = x.to(device)
                with torch.no_grad():
                    score = model(x, c)
                score = torch.sigmoid(score)
                y_pred = torch.where(score > 0.5, 1, 0)
                y_pred = y_pred.cpu()
                f1 = f1_score(y, y_pred, zero_division=1)
                recall = recall_score(y, y_pred, zero_division=1)
                precision = precision_score(y, y_pred, zero_division=1)
                f1_collect.append(f1)
                recall_collect.append(recall)
                precision_collect.append(precision)
                bar.update(1)
                bar.set_postfix(f1=np.mean(f1_collect), recall=np.mean(recall_collect),
                                precision=np.mean(precision_collect), best_f1=best_f1)
        if np.mean(f1_collect) > best_f1:
            best_f1 = np.mean(f1_collect)
            best_precision = np.mean(precision_collect)
            bast_recall = np.mean(recall_collect)

        best_f1_list.append(best_f1)
        if epoch > 10:
            if set(best_f1_list[-10:]) == {best_f1}:  # 10次f1不变
                # early stop and save model
                model_path = 'model/Dnn_{}_{}_v{}.pt'.format(time_span, category, epoch)
                torch.save(model.state_dict(), model_path)
                break

    print('----------------------------------')
    print('training finished')
    print('best f1: ', best_f1, 'best precision: ', best_precision, 'best recall: ', bast_recall)
    # print 2 train log
    with open(f'train_log_{time_span}_0805.txt', 'a') as f:
        f.write('category: {}\n'.format(category))
        f.write('best f1: {}\n'.format(best_f1))
        f.write('best precision: {}\n'.format(best_precision))
        f.write('best recall: {}\n'.format(bast_recall))
        f.write('stop epoch: {}\n'.format(len(loss_list)))
        f.write('----------------------------------\n')
    print('----------------------------------')


if __name__ == '__main__':
    for category in range(8):
        args = parameter_parser()
        train(args, time_span=args.time_span, version=0, category=category)
