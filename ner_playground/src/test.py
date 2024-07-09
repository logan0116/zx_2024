#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 下午11:44
# @Author  : liu yuhan
# @FileName: test.py
# @Software: PyCharm

import torch

a = torch.tensor([1, 2, 3])
# [1, 2, 3]->[1,1,2,2,3,3]
a = a.repeat_interleave(2)
print(a)


# input_shape = K.shape(inputs)
# batch_size, seq_len = input_shape[0], input_shape[1]
# position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]
#
# pos = sinusoidal_embeddings(position_ids, self.output_dim)
# qw, kw = apply_rotary_position_embeddings(pos, qw, kw)
#
#
# def sinusoidal_embeddings(pos, dim, base=10000):
#     """计算pos位置的dim维sinusoidal编码
#     """
#     assert dim % 2 == 0
#     indices = K.arange(0, dim // 2, dtype=K.floatx())
#     indices = K.pow(K.cast(base, K.floatx()), -2 * indices / dim)
#     embeddings = tf.einsum('...,d->...d', pos, indices)
#     embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
#     embeddings = K.flatten(embeddings, -2)
#     return embeddings
#
# def apply_rotary_position_embeddings(sinusoidal, *tensors):
#     """应用RoPE到tensors中
#     其中，sinusoidal.shape=[b, n, d]，tensors为tensor的列表，而
#     tensor.shape=[b, n, ..., d]。
#     """
#     assert len(tensors) > 0, 'at least one input tensor'
#     assert all([
#         K.int_shape(tensor) == K.int_shape(tensors[0]) for tensor in tensors[1:]
#     ]), 'all tensors must have the same shape'
#     ndim = K.ndim(tensors[0])
#     sinusoidal = align(sinusoidal, [0, 1, -1], ndim)
#     cos_pos = K.repeat_elements(sinusoidal[..., 1::2], 2, -1)
#     sin_pos = K.repeat_elements(sinusoidal[..., ::2], 2, -1)
#     outputs = []
#     for tensor in tensors:
#         tensor2 = K.stack([-tensor[..., 1::2], tensor[..., ::2]], ndim)
#         tensor2 = K.reshape(tensor2, K.shape(tensor))
#         outputs.append(tensor * cos_pos + tensor2 * sin_pos)
#     return outputs[0] if len(outputs) == 1 else outputs

#    def call(self, inputs, mask=None):
#         # 输入变换
#         inputs = self.dense(inputs)
#         inputs = tf.split(inputs, self.heads, axis=-1)
#         inputs = K.stack(inputs, axis=-2)
#         qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
#         # RoPE编码
#         if self.RoPE:
#             pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
#             qw, kw = apply_rotary_position_embeddings(pos, qw, kw)
#         # 计算内积
#         logits = tf.einsum('bmhd,bnhd->bhmn', qw, kw) / self.head_size ** 0.5
#         # 排除下三角
#         if self.tril_mask:
#             tril_mask = tf.linalg.band_part(K.ones_like(logits[0, 0]), 0, -1)
#             tril_mask = K.cast(tril_mask, 'bool')
#         else:
#             tril_mask = None
#         # 返回最终结果
#         return sequence_masking(logits, mask, -np.inf, [2, 3], tril_mask)

# import keras.backend as K
#
# position_ids = K.arange(0, 256, dtype=K.floatx())[None]
# print(position_ids)
# indices = K.arange(0, 64 // 2, dtype=K.floatx())
# indices = K.pow(K.cast(10000, K.floatx()), -2 * indices / 64)
# print(indices)
# # embeddings = tf.einsum('...,d->...d', pos, indices)
# # embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
# # embeddings = K.flatten(embeddings, -2)


def align(embeddings, axes, q_ndim=None):
    """重新对齐tensor（批量版expand_dims）
    axes：原来的第i维对齐新tensor的第axes[i]维；
    ndim：新tensor的维度。
    """
    indices = [None] * q_ndim
    for i in axes:
        indices[i] = slice(None)
    return embeddings[indices]


def get_rope(q):
    # [batch_size, head, seq_len, head_size]
    batch_size, head, seq_len, head_size = q.size()

    indices = torch.arange(0, head_size // 2, dtype=torch.float)
    indices = torch.pow(torch.tensor(10000, dtype=torch.float), -2 * indices / head_size)
    emb_cos = torch.cos(indices)
    emb_cos = torch.repeat_interleave(emb_cos, 2, dim=-1)
    emb_sin = torch.sin(indices)
    emb_sin = torch.repeat_interleave(emb_sin, 2, dim=-1)
    print(emb_cos, emb_sin)

    # [1, 1, 1, 1]->[-1,1,-1,1]
    trans_ = torch.Tensor([-1, 1] * (head_size // 2))
    print(q)
    q_ = q * trans_
    print(q_)
    return q * emb_cos + q_ * emb_sin

    # ndim = q.ndim
    # sinusoidal = align(embeddings, [0, 1, -1], ndim)
    # cos_pos = sinusoidal[..., 1::2].repeat_interleave(2, dim=-1)
    # sin_pos = sinusoidal[..., ::2].repeat_interleave(2, dim=-1)
    # outputs = []
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=4)
    q2 = q2.reshape(q.shape)
    # outputs.append(tensor * cos_pos + tensor2 * sin_pos)
    # return outputs

    #     indices = K.arange(0, dim // 2, dtype=K.floatx())
    #     indices = K.pow(K.cast(base, K.floatx()), -2 * indices / dim)
    #     embeddings = tf.einsum('...,d->...d', pos, indices)
    #     embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
    #     embeddings = K.flatten(embeddings, -2)

    #     ndim = K.ndim(tensors[0])
    #     sinusoidal = align(sinusoidal, [0, 1, -1], ndim)
    #     cos_pos = K.repeat_elements(sinusoidal[..., 1::2], 2, -1)
    #     sin_pos = K.repeat_elements(sinusoidal[..., ::2], 2, -1)
    #     outputs = []
    #     for tensor in tensors:
    #         tensor2 = K.stack([-tensor[..., 1::2], tensor[..., ::2]], ndim)
    #         tensor2 = K.reshape(tensor2, K.shape(tensor))
    #         outputs.append(tensor * cos_pos + tensor2 * sin_pos)
    #     return outputs[0] if len(outputs) == 1 else outputs


# q = torch.randn(1, 2, 4, 8)
# q = get_rope(q)
# print(q)
#
# # [-2.2325, -0.8883, -1.3463, -0.9649, -0.9546, -1.8106,  0.7747, -0.2291]
#
#
# import tensorflow as tf
#
# inputs = tf.random.normal([2, 3, 16 * 2])  # [batch_size, seq_len, head_size * 2]
# q_dense = tf.keras.layers.Dense(2 * 4, use_bias=False)
#
# logits = tf.einsum('bnh,bmh->bnm', inputs, inputs)  # [batch_size, seq_len, seq_len]
# print(logits.shape)
# print(logits[:, None].shape)
#
# bias = tf.einsum('bnh->bhn', q_dense(inputs)) / 2  # [batch_size, num_heads, seq_len]
# print(bias.shape)
# print(bias)
# print(bias[:, ::2, None].shape)
# print(bias[:, ::2, None])
# print(bias[:, 1::2, :, None].shape)
#
# logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
# print(logits.shape)
# # 排除下三角