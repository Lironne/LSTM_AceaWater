import glob
import os
import numpy as np
import pandas as pd
	
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

dirname = os.path.dirname(__file__)
data_path = os.path.join(dirname, 'data/*.csv')


def parse(x):
	return datetime.strptime(x, '%d/%m/%Y')


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def load_data():

    data = {}

    for data_set in glob.glob(data_path):

        path, name = os.path.split(data_set)
        name = os.path.splitext(name)[0]
        name = name.split('_')

        water_type = ''
        water_name = ''

        if(len(name) == 3):
            water_type = name[0]
            water_name = name[1]
        else:
            water_type = name[1]
            water_name = name[2]

        if(water_type not in data):
            data[water_type] = {}
        
        data[water_type][water_name] = pd.read_csv(data_set, parse_dates = ['Date'], index_col=0, date_parser=parse)

    return data 

def drop_cols(data, water_type, water_name):

    print(water_type, water_name)

    return data

def process_data(data):

    for water_type in data:
        for water_name in data[water_type]:

            dataset = data[water_type][water_name]
            values = dataset.values
            values = values.astype('float32')
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(values)
            reframed = series_to_supervised(scaled, 1, 1)
            data[water_type][water_name] = drop_cols(reframed, water_type, water_name)

    return data
