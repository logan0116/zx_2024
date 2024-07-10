#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：zx_2 
@File    ：model.py
@IDE     ：PyCharm 
@Author  ：Logan
@Date    ：2023/12/3 下午8:03 
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim, d_k, n_heads):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.d_k = d_k
        self.d_v = d_k
        self.n_heads = n_heads
        self.W_Q = nn.Linear(self.dim, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.dim, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.dim, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.dim)

    def get_attn(self, Q, K, V, attention_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # scores : [B, H, S, S]
        if attention_mask is not None:
            scores = self.scores_mask(scores, attention_mask)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context

    def scores_mask(self, scores, attention_mask):
        # scores: [batch_size, num_heads, seq_len, seq_len]
        # attention_mask: [batch_size, seq_len]
        # low_tri_mask: [seq_len, seq_len]
        # 1. attention_mask的mask
        attention_mask = attention_mask[:, None, None, :]  # [batch_size, 1, 1, seq_len]
        # 2. low_tri_mask的mask
        low_tri_mask = (1 - torch.tril(torch.ones(scores.size()[-2:]), diagonal=0)).to(
            scores.device)  # [seq_len,seq_len]
        # 3. mask combine
        mask = attention_mask + low_tri_mask - 1
        mask = torch.clamp(mask, min=0)
        mask = (1.0 - mask) * -1e12  # [batch_size, 1, seq_len, seq_len]
        return scores + mask

    def forward(self, input_Q, input_K, input_V, attention_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # context: [batch_size, n_heads, len_q, d_v]
        context = self.get_attn(Q, K, V, attention_mask)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        output = self.layer_norm(output + residual)
        return output


class MyDnn(nn.Module):
    def __init__(self):
        """
        2分类
        """
        super(MyDnn, self).__init__()
        # input

        self.fc_inputs = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.fc_categrory = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        # attention
        # self.attention = SelfAttention(dim=1024, d_k=128, n_heads=8)
        # output
        self.outputs = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.my_loss = nn.CrossEntropyLoss()

    def forward(self, x, c):
        # x: [batch_size, 1024] -> [batch_size, 1024]
        x = self.fc_inputs(x)
        # c: [8, 1024] -> [8, 1024]
        c = self.fc_categrory(c)
        # x_c: x - c [batch_size, 8， 1024]
        x_c = x.unsqueeze(1) - c.unsqueeze(0)
        # attention
        # x_c = self.attention(x_c, x_c, x_c, attention_mask=None)
        # x_c: [batch_size, 8， 1024] -> [batch_size, 8192]
        x_c = x_c.view(x_c.size(0), -1)
        # output
        output = self.outputs(x_c)
        return output

    def fun_loss(self, output, label):
        """
        :param output: [batch_size, 2]
        :param label: [batch_size]
        :return:
        """
        loss = self.my_loss(output, label)
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        loss += l2_lambda * l2_norm
        return loss


class MyDnnSimple(nn.Module):
    def __init__(self):
        """
        2分类
        """
        super(MyDnnSimple, self).__init__()
        # input

        self.fc_inputs = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.fc_category = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        # attention
        # self.attention = SelfAttention(dim=1024, d_k=128, n_heads=8)
        # output
        self.outputs = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x, c):
        # x: [batch_size, 1024]
        x = self.fc_inputs(x)
        # c: [1024]
        c = self.fc_category(c.unsqueeze(0))
        # output
        # norm
        x = F.normalize(x, p=2, dim=1)
        c = F.normalize(c, p=2, dim=1)
        output = self.outputs(x - c)
        return output


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
