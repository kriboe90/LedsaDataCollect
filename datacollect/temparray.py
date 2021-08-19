import numpy as np
import pandas as pd
import codecs

class TempArrayData:
    def __init__(self, path_temparray: str):
        self.window = 1
        self.smooth = 'ma'

        with codecs.open(path_temparray, 'r', encoding='utf-8', errors='ignore') as fdata:
            self.temparray_input_df = pd.read_csv(fdata, header=None, delimiter=';')

        self.__read_data()

    def __read_data(self):
        self.temparray_df = self.temparray_input_df.copy()
        self.temparray_df.dropna(axis=1, how='all', inplace=True)
        self.temparray_df.set_index([0], inplace=True)
        self.temparray_df.index.set_names(['Time'], inplace=True)
        self.temparray_df.rename(columns={1: "Signal", 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9},
                                 inplace=True)
        self.temparray_df.index = pd.to_timedelta(self.temparray_df.index - self.temparray_df.index.min(),
                                                  unit='ms').total_seconds()
        if self.smooth == 'median':
            self.temparray_df = self.temparray_df.iloc[::-1].rolling(window=self.window, closed='left').median().iloc[::-1]
        else:
            self.temparray_df = self.temparray_df.iloc[::-1].rolling(window=self.window, closed='left').mean().iloc[::-1]


    def smooth_data(self, window: int, smooth='ma'):
        self.window = window
        self.smooth = smooth
        self.__read_data()

    def set_timeshift(self, timedelta:int):
        self.temparray_df.index += timedelta
