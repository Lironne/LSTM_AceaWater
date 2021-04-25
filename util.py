import glob
import os
import numpy as np
import pandas as pd
	
from datetime import datetime
from dateutil.relativedelta import relativedelta
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

    dataset = pd.DataFrame(data)

    if water_type == 'Aquifer':
        if water_name == 'Auser':
            dataset.drop(dataset.columns[[26,27,28,29,30,31,32,33,34,35,38,40,41,42,43,44,45,46,47,48,49,50,51]], axis =1, inplace=True)
        elif water_name == 'Doganella':
            dataset.drop(dataset.columns[[21,22,32,33,34,35,36,37,38,39,40,41]], axis =1, inplace=True)
        elif water_name == 'Luco':
            dataset.drop(dataset.columns[[21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41]], axis =1, inplace=True)
        else:
            dataset.drop(dataset.columns[[7,10,11,12,13]], axis =1, inplace=True)
    elif water_type == 'Lake':
        dataset.drop(dataset.columns[[8,9,10,11,12,13]], axis =1, inplace=True)
    elif water_type == 'River':
        dataset.drop(dataset.columns[[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]], axis =1, inplace=True)
    else :
        if water_name == 'Amiata':
            dataset.drop(dataset.columns[[15,16,17,18,19,20,21,22,23,24,25]], axis =1, inplace=True)
        elif water_name == 'Lupa':
            dataset.drop(dataset.columns[[2]], axis =1, inplace=True)
        else:
            dataset.drop(dataset.columns[[3,4]], axis=1, inplace=True)

    return dataset

def process_data(data):

    for water_type in data:
        for water_name in data[water_type]:

            date_indices = data[water_type][water_name].index
            date_indices = date_indices.delete(date_indices.size - 1)
            dataset = data[water_type][water_name]
            values = dataset.values
            values = values.astype('float32')
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(values)
            reframed = series_to_supervised(scaled, 1, 1)
            data[water_type][water_name] = drop_cols(reframed, water_type, water_name)
            data[water_type][water_name].index = date_indices

    return data

def get_split_date(index):

    start_date = index[0]
    end_date = index[index.size - 1]

    if index.size < 365:
        return (round(index.size * 0.7) ,round(index.size * 0.85))

    if index.size < 365 * 3:
        split_date_test = end_date + relativedelta(months=-6)
        split_date_val = end_date + relativedelta(years=-1)

        split_idx_test = index.get_loc(split_date_test, method='nearest')
        split_idx_val = index.get_loc(split_date_val, method='nearest')

        return (split_idx_val,split_idx_test) 
    
    df = pd.DataFrame({'year': [end_date.year - 1, end_date.year - 2], 'month': [start_date.month, start_date.month], 'day': [start_date.day, start_date.day]})
    split_date = pd.to_datetime(df)

    split_idx_test = index.get_loc(split_date[0])
    split_idx_val = index.get_loc(split_date[1])

    return (split_idx_val,split_idx_test)


def get_output_dim(columns):

    output_vars = columns[columns.str.contains('(t)', regex=False)]

    return output_vars.size * -1
