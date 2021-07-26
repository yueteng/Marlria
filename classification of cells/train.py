
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model as keras_load_model
from keras import backend as K
from model.multi_scale_transformer import multi_scale_transformer
from utils.attention import AttentionLayer
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=tf_config)
def set_parameters():
    parser = argparse.ArgumentParser(description="Malaria Detection")
    parser.add_argument('--dataset', type=str, default='dataset1')
    parser.add_argument('--train_mode', type=str, default='train', )
    parser.add_argument('--model_save_path', type=str, default='./output/')
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--seed', type=int, default=2020)

    parser.add_argument('--model', type=str, default='AIM')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch_num', type=int, default=1000)
    parser.add_argument('--stop_num', type=int, default=100)
    parser.add_argument('--optimer', type=str, default="adam")


    parser.add_argument('--pretrain', type=bool, default=False)

    config = parser.parse_args()
    return config

def load_data(config):
    """ loading the sample data """
    path = f'./data/{config.dataset}'
    train_X = np.load(os.path.join(path, "train_data.npy"))
    train_Y = np.load(os.path.join(path, "train_label.npy"))

    datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, rotation_range=20,
                                 horizontal_flip=True, vertical_flip=True, brightness_range=[0.5, 1])
    train_generator = datagen.flow(train_X, train_Y, batch_size=config.batch_size)
    config.train_num = train_X.shape[0]

    valid_X = np.load(os.path.join(path, "valid_data.npy"))
    valid_Y = np.load(os.path.join(path, "valid_label.npy"))
    datagen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = datagen.flow(valid_X, valid_Y, batch_size=1)
    config.valid_num = valid_X.shape[0]

    test_X = np.load(os.path.join(path, "test_data.npy"))
    test_Y = np.load(os.path.join(path, "test_label.npy"))
    test_generator = datagen.flow(test_X, test_Y, batch_size=1)
    config.test_num = test_X.shape[0]


    return {"train": train_generator, "valid": valid_generator, "test": test_generator,
            "valid_X":valid_X, "valid_Y":valid_Y, "test_X": test_X, "test_Y": test_Y}



def load_model(config):
    models = {
        'AIM': multi_scale_transformer(),
    }
    assert config.model in models

    return models[config.model]


def train(config):
    """ Training Model """


    dataset = load_data(config)

    model = load_model(config)
    print(model.summary())

    model_save_path = f"{config.model_save_path}/{config.dataset}_{config.train_mode}_{config.model}.h5"
    early_stopping = EarlyStopping(monitor='val_acc', patience=config.stop_num, verbose=2, mode='max')
    check_point = ModelCheckpoint(model_save_path, monitor='val_acc', mode='max', verbose=2, save_best_only=True)
    callbacks_list = [check_point, early_stopping]

    #training the model
    history = model.fit_generator(dataset['train'],
                                  steps_per_epoch=config.train_num/config.batch_size,
                                  epochs=config.max_epoch_num,
                                  validation_data=dataset['valid'],
                                  validation_steps=config.valid_num,
                                  callbacks=callbacks_list, verbose=2)
    #Plotting Training and Testing accuracies


def evaluate(config):
    dataset = load_data(config)
    valid_X = dataset["valid_X"]
    valid_Y = dataset["valid_Y"]

    model_save_path = f"{config.model_save_path}/{config.dataset}_{config.train_mode}_{config.model}_{str(config.fold)}.h5"
    model = keras_load_model(model_save_path, custom_objects={'tf': tf, 'stack': K.stack,
                                                              'AttentionLayer': AttentionLayer})


    # obtaining accuracy on test set
    val_acc = model.evaluate(valid_X/255.0, valid_Y, verbose=0)

    print(model.metrics_names)
    print('Valid Accuracy Obtained: ')
    print(val_acc[1] * 100, ' %')

    test_X = dataset["test_X"]
    test_Y = dataset["test_Y"]
    test_acc = model.evaluate(test_X / 255.0, test_Y, verbose=2)


    print('Test Accuracy Obtained: ')
    print(test_acc[1] * 100, ' %')

if __name__ == "__main__":
    config = set_parameters()


    print(config)
    train(config)
    evaluate(config)

