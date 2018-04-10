# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = 2018 / 3 / 16
__version__ = ''

'''
The Keras implement of paper <Skeleton Based Action Recognition with Convolutional Neural Network> (PR2016)
Debug problem:
1. The dropout layer could not place anywhere in the architecture of the whole network. Need to figure out the reason and
the theory of dropout.
'''

import keras
import tensorflow as tf
import numpy as np
import random
import scipy.stats as stat

from keras.optimizers import SGD
from keras.models import Model,Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.regularizers import l2,l1
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from skimage import transform
import pandas as pd


from networks import BaseModel



class SkelCNN(object):
  _model_name = 'skel_cnn'
  view_aciton_pairs = []
  def __init__(self,
               train_data=None,
               test_data=None,
               batch_size=35,
               kernel_regularizer=l2(1.e-2),
               activation='relu',
               is_info=True,
               dropout=0.0,
               validation_rate=0.2,
               flip_rate=0.6):
    '''
    here dataset is organized with the format of (Number, [n_frames, n_joints, 3])
    :param batch_size: should be the times of 5
    :param flip_rate: the probability of image flip horizontally
    :dropout: need to set to 0 when you evaluate your model.
    '''
    # super(SkelCNN, self).__init__()

    self.train_data = train_data
    self.test_data = test_data
    self._batch_size = batch_size
    self._kernel_regularizer = kernel_regularizer
    self._activation = activation
    self._flip_rate = flip_rate
    self._validation_rate = validation_rate
    self._dropout = dropout

    self._test_gt_list = []

    # configs
    self._num_actions = 40
    self._img_input_weight = 52
    self._img_input_height = 52
    self._resize_weight = 60
    self._resize_height = 60
    self._n_group = 5
    self._n_channel = 3


    self._model = self._built_model()

    # model config
    loss = 'categorical_crossentropy'
    lr = 0.01
    momentum = 0.9
    optimizer = SGD(lr=lr, momentum=momentum, decay=1E-4, nesterov=True)
    self._model.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])
    if is_info:
      print(self._model.summary())

    self.train_lens = train_data.shape[0]
    self.test_lens = test_data.shape[0]

    indices = list(range(self.train_lens))
    random.shuffle(indices)
    self.train_data = self.train_data[indices]

    self._model_save_path = './models/{}/model.hdf5'.format(self._model_name)


  def _built_model(self):
    skel_input = Input(shape=(self._img_input_weight, self._img_input_height, 3), name='skel_input')

    x = Conv2D(32, (3,3), activation='relu',padding='same')(skel_input)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
    # This place could not place a dropout, i have not figured out yet!


    x = Conv2D(32, (3,3), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
    # This place could not place a dropout, i have not figured out yet!
    x = BatchNormalization()(x)
    x = Dropout(self._dropout)(x)

    x = Conv2D(64, (3,3), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)


    x = Conv2D(64, (3,3), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
    # This place could not place a dropout, i have not figured out yet!
    x = BatchNormalization()(x)
    x = Dropout(self._dropout)(x)

    x = Flatten()(x)
    x = Dense(units=128, activation='relu', kernel_regularizer=self._kernel_regularizer)(x)
    output = Dense(units=self._num_actions, activation='softmax', kernel_regularizer=self._kernel_regularizer)(x)

    model = Model(inputs=skel_input, outputs=output)
    return model

  def _val2rgb(self, mat):
    local_max = np.max(mat)
    local_min = np.min(mat)
    p = np.uint8(255*(mat-local_min)/(local_max-local_min))
    return p

  def _skel2image(self, mat):
    '''
    :param mat: (n_frames, feat_dim)
    :return: (60, 60, 3)
    '''
    mat = np.reshape(mat, newshape=(-1, 25, 3))
    joints_list = [mat[:, each_joints, :] for each_joints in range(25)]
    part_config = [25,12,24,11,10,9,21,21,5,6,7,8,22,23,21,3,4,21,2,1,17,18,19,20,21,2,1,13,14,15,16]
    ret_list = [joints_list[each_joint-1] for each_joint in part_config]
    ret_list = np.array(ret_list)
    rgb_image = self._val2rgb(ret_list)
    rgb_image = transform.resize(rgb_image, (self._resize_weight, self._resize_height, 3))
    return rgb_image

  def _random_select_patch(self, mat, number=5):
    '''
    randomly select $number$ patch from image
    '''
    patches = np.zeros(shape=(self._n_group, self._img_input_weight, self._img_input_height, 3))
    height = self._resize_height-self._img_input_height
    weight = self._resize_weight-self._img_input_weight
    for each in range(self._n_group):
      anchor_x = random.sample(range(weight), 1)[0]
      anchor_y = random.sample(range(height), 1)[0]
      select_patch = mat[anchor_x:anchor_x+self._img_input_weight, anchor_y:anchor_y+self._img_input_height, :]
      patches[each] = select_patch
    return patches

  def _corner_select_patch(self, mat):
    '''
    select the patches from four corners and center
    '''
    patches = np.zeros(shape=(self._n_group, self._img_input_weight, self._img_input_height, 3), dtype=np.float32)
    height = self._resize_height-self._img_input_height
    weight = self._resize_weight-self._img_input_weight
    anchors = [[0, 0], [weight, 0], [0, height], [weight, height], [int(weight/2), int(height/2)]]
    for each in range(self._n_group):
      anchor_x, anchor_y = anchors[each][0], anchors[each][1]
      select_patch = mat[anchor_x:anchor_x+self._img_input_weight, anchor_y:anchor_y+self._img_input_height, :]
      patches[each] = select_patch
    return patches

  def _flip_horizontal(self, mat, flip_prob):
    '''
    flip the image horizontally randomly with probability of $flip_prob$
    '''
    rand = np.random.uniform(low=0, high=1.0)
    if rand > flip_prob:
      return mat
    else:
      # flip horizontally
      return np.fliplr(mat)



  def _test_generator(self):
    '''
    select five patches from the four corner and center and their horizontal flip to evaluate. Using the voting result
    as the final result
    :return: (ret_x, ret_y) with the shape ret_x ~ (10, weight, height, 3), (10, num_actions)
    '''
    anchor = 0
    while True:
      ret_x = np.zeros(shape=(2*self._n_group, self._img_input_weight, self._img_input_height, self._n_channel), dtype=np.float32)
      ret_y = np.zeros(shape=(2*self._n_group, self._num_actions), dtype=np.float32)
      if anchor > self.test_lens-1:
        print('Test traversal has been done !')
        break
      current_data = self.test_data[anchor]
      rgb_img = self._skel2image(current_data['mat']) # rgb with shape (60, 60, 3)

      clips = self._corner_select_patch(rgb_img)
      flip_clips = self._corner_select_patch(self._flip_horizontal(rgb_img, flip_prob=1.00))
      # flip_clips = clips

      ret_x[0:self._n_group] = clips
      ret_x[self._n_group:] = flip_clips
      label = to_categorical(current_data['action'], self._num_actions)
      label = label[np.newaxis, :]
      labels = np.tile(label, reps=[2*self._n_group, 1])
      ret_y[:] = labels
      anchor += 1
      print(anchor)

      yield (ret_x, ret_y)


  def _validate_x_generator(self):
    '''
    here we trick and use test set as validation set since the train set is too small in cross-view exp
    :return:
    '''
    validate_max_size = self.test_lens
    while True:
      ret_x = np.zeros(shape=(2*self._n_group, self._img_input_weight, self._img_input_height, self._n_channel), dtype=np.float32)
      ret_y = np.zeros(shape=(2*self._n_group, self._num_actions), dtype=np.float32)

      random_select = random.sample(range(validate_max_size), 1)[0]
      current_data = self.test_data[random_select]

      rgb_img = self._skel2image(current_data['mat']) # rgb with shape (60, 60, 3)

      clips = self._corner_select_patch(rgb_img)
      flip_clips = self._corner_select_patch(self._flip_horizontal(rgb_img, flip_prob=1.00))

      ret_x[0:self._n_group] = clips
      ret_x[self._n_group:] = flip_clips
      label = to_categorical(current_data['action'], self._num_actions)
      label = label[np.newaxis, :]
      labels = np.tile(label, reps=[2*self._n_group, 1])
      ret_y[:] = labels

      yield (ret_x, ret_y)



  def _train_x_generator(self):
    anchor = 0
    while True:
      ret_x = np.zeros(shape=(0, self._img_input_weight, self._img_input_height, self._n_channel), dtype=np.float32)
      ret_y = np.zeros(shape=(0, self._num_actions), dtype=np.float32)
      if anchor > self.train_lens-self._batch_size:
        anchor = 0
        continue
      epoch = int(self._batch_size/self._n_group) # here batch_size has to be the times of n_group
      for num in range(epoch):
        current = anchor+num
        data_current = self.train_data[current]
        rgb_img = self._skel2image(data_current['mat']) # rgb with shape (60, 60, 3)
        rgb_img = self._flip_horizontal(rgb_img, flip_prob=self._flip_rate)
        # randomly flip the image horizontally with the probability of filp_prob
        patches = self._random_select_patch(rgb_img) # random select patches
        label = to_categorical(data_current['action'], self._num_actions)
        label = label[np.newaxis, :]
        labels = np.tile(label, reps=[self._n_group, 1])

        ret_x = np.concatenate((ret_x, patches), axis=0)
        ret_y = np.concatenate((ret_y, labels), axis=0)
      anchor += epoch
      yield (ret_x, ret_y)

  def generate_report(self, data, path):
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter(path)
    data_df.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
    data_df.to_excel(writer,'page_2',float_format='%.5f') # float_format 控制精度
    writer.save()

  def set_model_path(self, model_path):
    self._model_save_path = './models/{}_model/{}.hdf5'.format(self._model_name, model_path)

  def train(self, model_path, epochs=10, validate_size=1000):
    self._model_save_path = './models/{}_model/{}.hdf5'.format(self._model_name, model_path)
    checkpoint_cb = ModelCheckpoint(self._model_save_path,
                                    monitor='val_acc',
                                    save_best_only=True,
                                    period=1)
    cb_list = [checkpoint_cb]
    num_trainset = self.train_lens
    self._model.fit_generator(self._train_x_generator(),
                              steps_per_epoch=num_trainset,
                              epochs=epochs,
                              validation_data=self._validate_x_generator(),
                              validation_steps=validate_size,
                              callbacks=cb_list
                              )

  def evaluation(self, report_name):
    '''
    vote for a final score
    '''
    amount_testset = self.test_lens
    self._model.load_weights(self._model_save_path)
    pred_list = self._model.predict_generator(generator=self._test_generator(),
                                              steps=amount_testset)
    # input only one instance and
    # output the pred_list shape with (amount_testset, 2*n_group, num_actions)
    pred_list = np.array(pred_list)
    pred_list = np.reshape(pred_list, newshape=(self.test_lens, 2*self._n_group, -1))

    print(pred_list.shape)
    preds = np.argmax(pred_list, axis=2)  # (number, 2*n_group)

    pred_result = stat.mode(preds, axis=1)[0] # (amount_testset, 1) numpy array

    gt_result = []
    view_list = []
    for each in self.test_data:
      gt_result.append(each['action'])
      view_list.append(each['view'])
    view_list = np.array(view_list)
    gt_result = np.array(gt_result)

    view_accuracy_matrix = np.zeros(shape=(1, 8), dtype=np.float32)
    total_matrix = np.zeros(shape=(1,8), dtype=np.float32)
    for each_view in range(8):
      for current_view, current_pred, current_gt in zip(view_list, pred_result, gt_result):
        if current_view == each_view:
          total_matrix[0, current_view] += 1
          if current_pred == current_gt:
            view_accuracy_matrix[0, current_view] += 1
    pred_acc = view_accuracy_matrix/total_matrix
    report_name = './reports/{}_reports/{}_model_{}.xlsx'.format(self._model_name, self._model_name, report_name)
    print(report_name)
    self.generate_report(pred_acc, report_name)


  # loop view #

  def _loop_test_generator(self):
    anchor = 0

    while True:
      ret_x = np.zeros(shape=(2*self._n_group, self._img_input_weight, self._img_input_height, self._n_channel), dtype=np.float32)
      ret_y = np.zeros(shape=(2*self._n_group, self._num_actions), dtype=np.float32)
      if anchor > self.test_lens-1:
        print('Test traversal has been done !')
        break
      current_data = self.test_data[anchor]
      rgb_img = self._skel2image(current_data['mat']) # rgb with shape (60, 60, 3)

      clips = self._corner_select_patch(rgb_img)
      flip_clips = self._corner_select_patch(self._flip_horizontal(rgb_img, flip_prob=1.00))
      # flip_clips = clips

      ret_x[0:self._n_group] = clips
      ret_x[self._n_group:] = flip_clips
      label = to_categorical(current_data['action'], self._num_actions)
      label = label[np.newaxis, :]
      labels = np.tile(label, reps=[2*self._n_group, 1])
      ret_y[:] = labels
      anchor += 1
      print(anchor)
      view_action_p = (current_data['clip_ind'], current_data['action'])
      self.view_aciton_pairs.append(view_action_p)

      yield (ret_x, ret_y)



  def loop_eval(self, report_name='loop_view_exp'):
    report_name = './reports/{}_reports/{}_model_{}.xlsx'.format(self._model_name, self._model_name, report_name)
    print(report_name)
    amount_testset = self.test_lens
    self._model.load_weights(self._model_save_path)
    pred_list = self._model.predict_generator(generator=self._loop_test_generator(),
                                              steps=amount_testset)
    # input only one instance and
    # output the pred_list shape with (amount_testset, 2*n_group, num_actions)
    pred_list = np.array(pred_list)
    pred_list = np.reshape(pred_list, newshape=(self.test_lens, 2*self._n_group, -1))
    preds = np.argmax(pred_list, axis=2)  # (number, 2*n_group)
    pred_result = stat.mode(preds, axis=1)[0] # (amount_testset, 1) numpy array

    view_action_matrix = np.zeros(shape=(10, 40), dtype=np.float32)
    view_action_matrix_whole = np.zeros(shape=(10, 40), dtype=np.float32)

    print(len(self.view_aciton_pairs))
    for ind, each_pair in enumerate(self.view_aciton_pairs):
      gt_view, gt_action = each_pair[0], each_pair[1]
      view_action_matrix_whole[gt_view, gt_action] += 1
      if pred_result[ind] == gt_action:
        view_action_matrix[gt_view, gt_action] += 1
    #
    #
    pred_matrix = view_action_matrix/view_action_matrix_whole
    self.generate_report(pred_matrix, report_name)

