from pkgutil import get_data
import tensorflow as tf
import sonnet as snt
import numpy as np
import pandas as pd
from operator import add
from functools import reduce

np.set_printoptions(threshold=np.inf)

def build_ontar_model(inputs_sg, scope='ontar'):
    channel_size = [8, 32, 64, 64, 256, 256]
    betas = [None] + [tf.Variable(0.0 * tf.ones(channel_size[i]), name=f'beta_{i}') for i in
                        range(1, len(channel_size))]

    e1 = snt.Conv2D(channel_size[1], kernel_shape=[1, 3], name='e_1')
    ebn1u = snt.BatchNorm(decay_rate=0, create_scale=False,create_offset=False, name='ebn_1u')
    e2 = snt.Conv2D(channel_size[2], kernel_shape=[1, 3], stride=2, name='e_2')
    ebn2u = snt.BatchNorm(decay_rate=0, create_scale=False,create_offset=False, name='ebn_2u')
    e3 = snt.Conv2D(channel_size[3], kernel_shape=[1, 3], name='e_3')
    ebn3u = snt.BatchNorm(decay_rate=0, create_scale=False,create_offset=False, name='ebn_3u')
    e4 = snt.Conv2D(channel_size[4], kernel_shape=[1, 3], stride=2, name='e_4')
    ebn4u = snt.BatchNorm(decay_rate=0, create_scale=False,create_offset=False, name='ebn_4u')
    e5 = snt.Conv2D(channel_size[5], kernel_shape=[1, 3], name='e_5')
    ebn5u = snt.BatchNorm(decay_rate=0, create_scale=False,create_offset=False, name='ebn_5u')

    encoder = [None, e1, e2, e3, e4, e5]
    encoder_bn_u = [None, ebn1u, ebn2u, ebn3u, ebn4u, ebn5u]

    hu0 = inputs_sg
    u_lst = [hu0]
    hu_lst = [hu0]

    for i in range(1, len(channel_size) - 1):
        hu_pre = hu_lst[i - 1]
        pre_u = encoder[i](hu_pre)
        u = encoder_bn_u[i](pre_u, False, test_local_stats=False)
        hu = tf.nn.relu(u + betas[i])
        u_lst.append(u)
        hu_lst.append(hu)

    hu_m1 = hu_lst[-1]
    pre_u_last = encoder[-1](hu_m1)
    u_last = encoder_bn_u[-1](pre_u_last, False, test_local_stats=False)
    u_last = u_last + betas[-1]
    hu_last = tf.nn.relu(u_last)
    u_lst.append(u_last)
    hu_lst.append(hu_last)

    # classifier
    cls_channel_size = [512, 512, 1024, 2]
    e6 = snt.Conv2D(cls_channel_size[0], kernel_shape=[1, 3], stride=2, name='e_6')
    ebn6l = snt.BatchNorm(decay_rate=0.99, name='ebn_6l')
    e7 = snt.Conv2D(cls_channel_size[1], kernel_shape=[1, 3], name='e_7')
    ebn7l = snt.BatchNorm(decay_rate=0.99, name='ebn_7l')
    e8 = snt.Conv2D(cls_channel_size[2], kernel_shape=[1, 3], padding='VALID', name='e_8')
    ebn8l = snt.BatchNorm(decay_rate=0.99, name='ebn_8l')
    e9 = snt.Conv2D(cls_channel_size[3], kernel_shape=[1, 1], name='e_9')

    cls_layers = [None, e6, e7, e8, e9]
    cls_bn_layers = [None, ebn6l, ebn7l, ebn8l]

    hl0 = hu_last
    l_lst = [hl0]
    hl_lst = [hl0]
    for i in range(1, len(cls_channel_size)):
        hl_pre = hl_lst[i - 1]
        pre_l = cls_layers[i](hl_pre)
        l = cls_bn_layers[i](pre_l, False, test_local_stats=False)
        hl = tf.nn.relu(l)
        l_lst.append(l)
        hl_lst.append(hl)

    hl_m1 = hl_lst[-1]
    l_last = cls_layers[-1](hl_m1)
    hl_last = tf.nn.softmax(l_last)
    l_lst.append(l_last)
    hl_lst.append(hl_last)

    sig_l = tf.squeeze(hl_last, axis=[1, 2])[:, 1]
    print(sig_l.shape)
    return sig_l

# inputs_sg = tf.placeholder(dtype=tf.float32, shape=[None, 1, 23, 8])
inputs_sg = tf.keras.Input(shape=(1, 23, 8))
pred_ontar = build_ontar_model(inputs_sg)