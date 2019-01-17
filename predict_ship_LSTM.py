# -*- coding:UTF-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
# print(sys.path)
import csv
import datetime
import numpy as np
import pandas as pd
from statsmodels.tsa.tsatools import lagmat


def create_data_set(data_set, timestep=1, look_back=1, look_ahead=1):
    data_x = lagmat(data_set[:-look_ahead], maxlag=timestep*look_back, trim="both", original='ex')
    data_y = lagmat(data_set[(timestep*look_back):], maxlag=look_ahead, trim="backward", original='ex')
    data_x = data_x[:, ::-1].reshape(data_x.shape[0], timestep, look_back)[:-(look_ahead-1)]
    data_y = data_y[:, ::-1][:-(2*look_ahead-1)]

    return data_x, data_y


def data_preprocess(pyr, ori, norm, file, look_ahead, look_back, cutoff1, cutoff2):
    timestep = 1
    df = pd.read_csv(file, header=None, encoding='UTF-8', delim_whitespace=True, names=['x-s', 'y-s', 'z-s', 'p', 'y', 'r'])
    df = df[5000:170000]
    df1 = df[pyr]
    pitname = pyr + '.png'
    # df1.plot()
    # plt.savefig(pitname, format='png')
    train, test, val = df1[:cutoff1], df1[cutoff1:-cutoff2], df1[-cutoff2:]

    global trainX, trainY, testX, testY, valX, valY
    trainX, trainY = create_data_set(train, timestep=timestep, look_back=look_back, look_ahead=look_ahead)
    testX, testY = create_data_set(test, timestep=timestep, look_back=look_back, look_ahead=look_ahead)
    valX, valY = create_data_set(val, timestep=timestep, look_back=look_back, look_ahead=look_ahead)


def build_model(look_ahead, look_back):
    import tool.keras_pso.models as kModels
    import tool.keras_pso.layers as kLayers
    
    model = kModels.Sequential()
    model.add(kLayers.LSTM(256, input_shape=(1, look_back), kernel_initializer='he_uniform', activation='relu', return_sequences=True))
    model.add(kLayers.LSTM(256, dropout_W=0.2, dropout_U=0.2, kernel_initializer='he_uniform', activation='relu', return_sequences=True))
    model.add(kLayers.LSTM(256, dropout_W=0.2, dropout_U=0.2, kernel_initializer='he_uniform', activation='relu', return_sequences=True))
    model.add(kLayers.LSTM(256, dropout_W=0.2, dropout_U=0.2, kernel_initializer='he_uniform', activation='relu', return_sequences=False))
    model.add(kLayers.Dense(1024, activation='relu'))
    model.add(kLayers.Dropout(0.2))
    model.add(kLayers.Dense(look_ahead))

    return model


def train_optimize(pyr, model, batch_size, h5_file, optimizer='pso', epochs=100, err_best_g=1.0, adam=True, num_particles=10, psomaxiter=10):
    from tool.keras_pso.optimizers import Pso
    from tool.keras_pso.models import load_model
    from tool.keras_pso.callbacks import ModelCheckpoint, TensorBoard

    if os.path.exists(h5_file):
        model = load_model(h5_file)
    else:
        # if optimizer=='pso':
        #     optimize=Pso(adam, num_particles=num_particles, psomaxiter=psomaxiter)
        #     model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'], adam=adam)
        # else:
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(valX, valY),
                callbacks=[
                    ModelCheckpoint(filepath=h5_file, save_best_only=True, save_weights_only=False, monitor='val_loss'),
                    TensorBoard(log_dir='./log_' + optimizer, write_graph=True, write_images=True)])
    
    pred = model.predict(valX)
    mse = (((pred - valY) ** 2).sum(axis=1) / 300).sum() / pred.shape[0]          

    return model, mse


def optimize_model(err_best_g=0.0, model=None, h5_file=None, num_particles=10, psomaxiter=20, epochs=100, batch_size=32, optimize=True, initial_weights=None, adam=False):
    from tool.keras_pso.optimizers import Pso
    from tool.keras_pso.models import load_model
    from tool.keras_pso.callbacks import ModelCheckpoint, TensorBoard

    if initial_weights is not None and optimize is False:
        # model.summary()
        i = 0
        for layer in model.layers:
            if 'lstm' in layer.name and len(layer.weights) > 0:
                print('initial_weights:', [initial_weights[i], initial_weights[i+1], initial_weights[i+2]])
                value = np.asarray(initial_weights[i])
                print('value: ', value)
                layer.set_weights([initial_weights[i], initial_weights[i+1], initial_weights[i+2]])
                i = i + 3
            elif 'dense' in layer.name and len(layer.weights) > 0:
                layer.set_weights([initial_weights[i], initial_weights[i+1]])
                i = i + 2
        # losses = model.evaluate(valX, valY, batch_size=batch_size, verbose=1)
        # loss_sum = 0
        # for loss in losses:
        #     loss_sum += loss
        # loss = loss_sum / len(losses)
        # print('loss:', loss)
        # return loss
        pred = model.predict(valX)
        mse = (((pred - valY) ** 2).sum(axis=1) / 300).sum() / pred.shape[0]
        return mse         

    if os.path.exists(h5_file):
        print('h5_file:', h5_file)
        print('err_best_g:', err_best_g)
        model = load_model(h5_file)
    else:
        print('h5_file:', h5_file)
        print('err_best_g:', err_best_g)
        pso = Pso(err_best_g, num_particles=num_particles, psomaxiter=psomaxiter, optimize=optimize, adam=adam)
        model.compile(loss='mse', optimizer=pso, metrics=['accuracy'])
        start = datetime.datetime.now()
        model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(valX, valY),
                callbacks=[ModelCheckpoint(filepath=h5_file, monitor='val_loss'), TensorBoard(log_dir='./log_pso', write_graph=True, write_images=True)])
        end = datetime.datetime.now()
        print('end - start:(train time)', end - start)

    start = datetime.datetime.now()
    losses = model.evaluate(valX, valY, batch_size=batch_size, verbose=1)
    end = datetime.datetime.now()
    print('end - start:(test time)', end - start)

    # loss_sum = 0
    # for loss in losses:
    #     loss_sum += loss
    # loss = loss_sum / len(losses)
    # return loss
    pred = model.predict(valX)
    mse = (((pred - valY) ** 2).sum(axis=1) / 300).sum() / pred.shape[0]
    return mse         


def test_model(pyr, model, norm, ori, optimizer='adam'):
    pred = model.predict(testX)
    mse = (((pred - testY) ** 2).sum(axis=1) / 300).sum() / pred.shape[0]          
    title = pyr
    i = 0
    # for y, p in zip(testY, pred):
    # print(y.shape)
    # print(p.shape)
    y = testY[-5000, :]
    p = pred[-5000, :]
    plt.figure(i)
    plt.title(title)
    plt.xlabel('时间步长')
    if 's' in pyr:
        plt.ylabel(pyr+'(海里/小时)')
    else:
        plt.ylabel(pyr+'(°)')
    plt.plot(y, color='red', linestyle='-', linewidth=2.0, label='true test data')
    plt.plot(p, color='green', linestyle='--', linewidth=2.0, label='predict data')
    plt.legend(loc='best')
    pig = norm + '_' + ori + '_' + pyr + '_' + optimizer + '/' + str(datetime.date.today()) + '_' + str(i) +'.png'
    if not os.path.exists(norm + '_' + ori + '_' + pyr + '_' + optimizer + '/'):
        os.makedirs(norm + '_' + ori + '_' + pyr + '_' + optimizer + '/')
    plt.savefig(pig, format='png')
    i += 1

    return mse
