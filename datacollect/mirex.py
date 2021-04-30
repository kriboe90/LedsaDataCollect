import numpy as np
import pandas as pd
import codecs

class MirexData:
    def __init__(self, path_mirex: str):

        with codecs.open(path_mirex, 'r', encoding='utf-8', errors='ignore') as fdata:
            self.mirex_input_df = pd.read_csv(fdata, skiprows = 19, delimiter='\t')
            self.window = 1
            self.smooth = 'ma'
            self.__read_data()

    def __read_data(self):
        sigma_1 = lambda b : np.log(10**(b/10))
        sigma_2 = lambda b : 5*np.log10(1/b*10)/(np.log10(np.e)*10)
        if self.smooth == 'median':
            self.mirex_data_df = self.mirex_input_df.iloc[::-1].rolling(window=self.window, closed='left').median().iloc[::-1]
        else:
            self.mirex_data_df = self.mirex_input_df.iloc[::-1].rolling(window=self.window, closed='left').mean().iloc[::-1]
        self.real_time = self.mirex_data_df['# Time : ;  Sample']
        self.mirex_data_df.index = self.real_time.to_numpy()
        self.mirex_1 = self.mirex_data_df['MIREX    '].apply(sigma_1) # h = 3.30 m
        self.mirex_2 = self.mirex_data_df['Mirex 2  '].apply(sigma_2) # h = 2.30 m
        self.mirex_3 = self.mirex_data_df['Mirex 3  '].apply(sigma_2) # h = 1.52 m
        self.mass_loss_rate = self.mirex_data_df['Balance'].diff()

    def smooth_data(self, window: int, smooth='ma'):
        self.window = window
        self.smooth = smooth
        self.__read_data()
    def set_timeshift(self, timedelta:int):
        self.mirex_1.index += timedelta
        self.mirex_2.index += timedelta
        self.mirex_3.index += timedelta
#         self.mass_loss_rate.index += timedelta