import util
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def split_data(data):
    # split into train, val and test sets
    values = data.values
    split_date_val, split_date_test = util.get_split_date(data.index)
    train = values[:split_date_val, :]
    val = values[split_date_val:split_date_test,:]
    test = values[split_date_test:, :]
    # split into input and outputs
    output_dim = util.get_output_dim(data.columns)
    train_X, train_y = train[:, :output_dim], train[:, output_dim:]
    val_X, val_y = val[:,:output_dim], val[:,output_dim:]
    test_X, test_y = test[:, :output_dim], test[:, output_dim:]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return (train_X,train_y,val_X,val_y,test_X,test_y)

def gen_models(data):

    models = {} 

    for water_type in data:
        for water_name in data[water_type]: 

            if water_type not in models:
                models[water_type] = {}
            
            train_X,train_y,val_X,val_y,test_X,test_y = split_data(data[water_type][water_name])
            model = Sequential()
            model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dense(1))
            model.compile(loss='mae', optimizer='adam')
            # fit network
            history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(val_X, val_y), verbose=2, shuffle=False)
            models[water_type][water_name] = model
    return models