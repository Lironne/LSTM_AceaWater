import glob
import os
import numpy as np
import pandas as pd

dirname = os.path.dirname(__file__)
data_path = os.path.join(dirname, 'data/*.csv')


def load_data():

    data = {}

    for data_set in glob.glob(data_path):

        path, name = os.path.split(data_set)
        name = os.path.splitext(name)[0]
        name = name.split('_')

        water_type = ''
        water_name = ''

        if(len(name) == 2):
            water_type = name[0]
            water_name = name[1]
        else:
            water_type = name[1]
            water_name = name[2]

        if(water_type not in data):
            data[water_type] = {}
        
        data[water_type][water_name] = pd.read_csv(data_set)
    
    return data 

    

