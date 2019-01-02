# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function

import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
#from tensorflow.contrib.data import Iterator
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction
from util.data_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
from util.MSNet import MSNet

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        # add your own networks here
        print('unsupported network:', name)
        exit()


def topo_loss(pred, y, orig_y, weight_map, tumor):

	pred = tf.squeeze(pred)
	y = tf.squeeze(y)
	h,w,d = y.get_shape()
	gt = tf.one_hot(y, 2)

	orig_y = tf.squeeze(orig_y)
	o = tf.ones_like(orig_y)
	z = tf.zeros_like(orig_y)
	o_gt = tf.ones_like(gt)
	
	if tumor == 'tc':
		orig_y = tf.where(tf.equal(orig_y,4), o,orig_y)
		orig_y = tf.where(tf.equal(orig_y,2), o,orig_y)
	elif tumor == 'en':
		orig_y = tf.where(tf.equal(orig_y,4), o,orig_y)
		orig_y = tf.where(tf.equal(orig_y,2), z,orig_y)


	out = tf.argmax(pred, 3)

	invalid_map = out * (o - orig_y)
	invalid_map = tf.tile([invalid_map],[2,1,1,1])
	invalid_map = tf.transpose(invalid_map, [1,2,3,0])
	invalid_map = tf.cast(tf.reshape(invalid_map, [h,w,d,2]), tf.float32)
	orig_gt = tf.one_hot(orig_y, 2)

	loss_matrix = - 1 * (orig_gt * pred) * invalid_map 

	return tf.reduce_mean(loss_matrix)


def smooth_loss(pred, y, orig_y, weight_map, tumor):

	pred = tf.squeeze(pred)
	y = tf.squeeze(y)
	h,w,d = y.get_shape()
	gt = tf.one_hot(y, 2)


	z1 = tf.zeros_like(gt[1:,:,:,:])
	z2 = tf.zeros_like(gt[:,1:,:,:])
	z3 = tf.zeros_like(gt[:,:,1:,:])

	valid1 = tf.equal(gt[1:,:,:,:]  ,gt[:h-1,:,:,:])
	valid2 = tf.equal(gt[:,1:,:,:]  ,gt[:,:w-1,:,:])
	valid3 = tf.equal(gt[:,:,1:,:]  ,gt[:,:,:d-1,:])


	a =   tf.where(valid1, gt[1:,:,:,:] * tf.abs(pred[1:,:,:,:] - pred[:h-1,:,:,:]),z1)
	b =   tf.where(valid2, gt[:,1:,:,:] * tf.abs(pred[:,1:,:,:] - pred[:,:w-1,:,:]),z2)
	c =   tf.where(valid3, gt[:,:,1:,:] * tf.abs(pred[:,:,1:,:] - pred[:,:,:d-1,:]),z3)

	return (tf.reduce_mean(a) + tf.reduce_mean(b) + tf.reduce_mean(c))/3


def train(config_file):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']
     
    
    

    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])

    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    class_num   = config_net['class_num']
    batch_size  = config_data.get('batch_size', 5)
   
    # 2, construct graph
    full_data_shape  = [batch_size] + config_data['data_shape']
    full_label_shape = [batch_size] + config_data['label_shape']
    x = tf.placeholder(tf.float32, shape = full_data_shape)
    w = tf.placeholder(tf.float32, shape = full_label_shape)
    y = tf.placeholder(tf.int64,   shape = full_label_shape)
    orig_y = tf.placeholder(tf.int64,   shape = full_label_shape)
   
    w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(net_type)
    net = net_class(num_classes = class_num,
                    w_regularizer = w_regularizer,
                    b_regularizer = b_regularizer,
                    name = net_name)
    net.set_params(config_net)
    predicty = net(x, is_training = True)
    proby    = tf.nn.softmax(predicty)
    
    loss_func = LossFunction(n_class=class_num)
    loss = loss_func(predicty, y, weight_map = w)


    if net_name[6:8] == 'TC':
        loss =   loss + 3 * topo_loss(proby,y, orig_y, w, 'tc') + 3 * smooth_loss(proby,y, orig_y, w, 'tc')
    elif net_name[6:8] == 'EN':
        loss =   loss + 3 * topo_loss(proby,y, orig_y, w, 'en') + 3 * smooth_loss(proby,y, orig_y, w, 'en')
    elif net_name[6:8] == 'WT':
        loss = loss
    else:
        print ('Error')
        exit()



    print('size of predicty:',predicty)
    
    # 3, initialize session and saver
    lr = config_train.get('learning_rate', 1e-3)
    opt_step = tf.train.AdamOptimizer(1e-5,0.99,0.9999).minimize(loss)
    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer())  
    saver = tf.train.Saver()
    
    dataloader = DataLoader(config_data)
    dataloader.load_data()
    
    # 4, start to train
    loss_file = config_train['model_save_prefix'] + "_loss.txt"
    start_it  = config_train.get('start_iteration', 0)
    if( start_it > 0):
        saver.restore(sess, config_train['model_pre_trained'])
    loss_list, temp_loss_list = [], []
    for n in range(start_it, 40000):
        train_pair = dataloader.get_subimage_batch()
        tempx = train_pair['images']
        tempw = train_pair['weights']
        tempy = train_pair['labels']
        orig_tempy = train_pair['orig_labels']
        opt_step.run(session = sess, feed_dict={x:tempx, w: tempw, y:tempy, orig_y:orig_tempy})

        if(n%config_train['test_iteration'] == 0):
            batch_dice_list = []
            for step in range(config_train['test_step']):
                train_pair = dataloader.get_subimage_batch()
                tempx = train_pair['images']
                tempw = train_pair['weights']
                tempy = train_pair['labels']
                dice = loss.eval(feed_dict ={x:tempx, w:tempw, y:tempy, orig_y:orig_tempy})
                batch_dice_list.append(dice)
            batch_dice = np.asarray(batch_dice_list, np.float32).mean()
            t = time.strftime('%X %x %Z')
            print(t, 'n', n,'loss', batch_dice)
            loss_list.append(batch_dice)
            np.savetxt(loss_file, np.asarray(loss_list))

        if((n+1)%config_train['snapshot_iteration']  == 0):
            saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(n+1))
    sess.close()
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train.py config17/train_wt_ax.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    train(config_file)
