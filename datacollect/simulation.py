import os
import glob
import pandas as pd

class SimData:
    def __init__(self, path_simulation: str, all=True, average_images=False):
        self.path_simulation = path_simulation
        if average_images == True:
            image_infos_file = 'analysis/image_infos_analysis_avg.csv'
        else:
            image_infos_file = 'analysis/image_infos_analysis.csv'

        self.image_info_df = pd.read_csv(os.path.join(self.path_simulation, image_infos_file))

        if all == True:
            self.read_all()
        else:
            self.ch0_ledparams = None
            self.ch1_ledparams = None
            self.ch2_ledparams = None
            self.ch0_extcos = None
            self.ch1_extcos = None
            self.ch2_extcos = None

    height_from_layer = lambda self, layer: -1*(layer/self.n_layers*(self.top_layer-self.bottom_layer)-self.top_layer)
    layer_from_height = lambda self, height: int((self.top_layer-self.bottom_layer)/(self.top_layer - self.bottom_layer)*self.n_layers)

    def set_layer_params(self, bottom: float, top: float):
        """Set height above floor for lowest and highest layer"""
        self.bottom_layer = bottom
        self.top_layer = top

    def set_timeshift(self, timedelta: int):
        """Set timeshift [seconds] for the beginning of the experiment"""
        self.all_extco_df.index += timedelta

    def _get_ledparams_df_from_path(self, channel: int) -> pd.DataFrame:
        """Read binary hdf table and set experimental time as index"""
        file = os.path.join(self.path_simulation, 'analysis', f'channel{channel}', 'all_parameters.h5')
        table = pd.read_hdf(file, 'table')
        time = self.image_info_df['Experiment_Time[s]'].astype(int)
        table = table.merge(time, left_on='img_id', right_index=True)
        table.set_index(['Experiment_Time[s]', 'led_id'], inplace=True)
        self.led_heights = table['height']
        return table

    def _get_extco_df_from_path(self):
        """
        Read all extinction coefficients from the simulation dir and put them in the all_extco_df.
        Get number of layers found in the csv.
        """
        extco_list = []
        files_list = glob.glob(os.path.join(self.path_simulation, 'analysis/AbsorptionCoefficients/', f'absorption_coefs*.csv'))
        for file in files_list:
            file_df = pd.read_csv(file, skiprows=4)
            channel = int(file.split('channel_')[1].split('_')[0])
            line = int(file.split('array_')[1].split('.')[0])
            n_layers = len(file_df.columns)
            time = self.image_info_df['Experiment_Time[s]'].astype(int)
            file_df = file_df.merge(time, left_index=True, right_index=True)
            file_df.set_index('Experiment_Time[s]', inplace=True)
            iterables = [[channel],[line], [i for i in range(0,n_layers)]]
            file_df.columns = pd.MultiIndex.from_product(iterables, names = ["Channel", "Line", "Layer"])
            extco_list.append(file_df)
        self.all_extco_df = pd.concat(extco_list, axis=1)
        self.all_extco_df.sort_index(ascending=True, axis=1, inplace=True)
        self.n_layers = n_layers

    def read_all(self):
        """Read led parameters and extionciton coefficients for all color channels from the simulation path"""
        self.ch0_ledparams_df = self._get_ledparams_df_from_path(0)
        self.ch1_ledparams_df = self._get_ledparams_df_from_path(1)
        self.ch2_ledparams_df = self._get_ledparams_df_from_path(2)
        self._get_extco_df_from_path()

    def get_extco_at_timestep(self, channel: int, timestep: int, yaxis='layer', window=1, smooth='ma') -> pd.DataFrame:
        """
        Get a Dataframe containing extinction coefficients at timestep.
        Index = Layer, Columns = Line
        Smooth over time (window) by moving average (ma) or meadian (meadian)
        """
        ch_extco_df = self.all_extco_df.xs(channel, level=0, axis=1)
        if smooth == 'median':
            ma_ch_extco_df = ch_extco_df.iloc[::-1].rolling(window=window, closed='left').median().iloc[::-1]
        else:
            ma_ch_extco_df = ch_extco_df.iloc[::-1].rolling(window=window, closed='left').mean().iloc[::-1]
        ma_ch_extco_df = ma_ch_extco_df.loc[timestep, :]
        ma_ch_extco_df = ma_ch_extco_df.reset_index().pivot(columns='Line',index='Layer')
        ma_ch_extco_df.columns = ma_ch_extco_df.columns.droplevel()
        # print(ma_ch_extco_df)
        ma_ch_extco_df.index = range(ma_ch_extco_df.shape[0])
        if yaxis == 'layer':
            ma_ch_extco_df.index.names = ["Layer"]
        elif yaxis == 'height':
            ma_ch_extco_df.index = [self.height_from_layer(layer) for layer in ma_ch_extco_df.index]
            ma_ch_extco_df.index.names = ["Height"]
        #
        # ma_ch_extco_df.columns = range(ma_ch_extco_df.shape[1])
        # ma_ch_extco_df.columns.names = ["Line"]
        return ma_ch_extco_df

    def get_extco_at_line(self, channel: int, line: int,yaxis='layer', window=1) -> pd.DataFrame:
        """
        Get a Dataframe containing extinction coefficients at line.
        Index = experimental time, Columns = Layer
        Smooth over time (window) by moving average (ma) or meadian (meadian)
        """
        ch_extco_df = self.all_extco_df.xs(channel, level=0, axis=1).xs(line, level=0, axis=1)
        ma_ch_extco_df = ch_extco_df.rolling(window=window, closed='right').mean().shift(-int(window/2)+1)
        # ma_ch_extco_df = ch_extco_df.iloc[::-1].rolling(window=window, closed='left').mean().iloc[::-1]

        if yaxis == 'layer':
            ma_ch_extco_df.columns.names = ["Layer"]
        elif yaxis == 'height':
            ma_ch_extco_df.columns = [self.height_from_layer(layer) for layer in ma_ch_extco_df.columns]
            ma_ch_extco_df.columns.names = ["Height"]
        return ma_ch_extco_df

    def get_extco_at_layer(self, channel: int, layer: int, window=1) -> pd.DataFrame:
        """
        Get a Dataframe containing extinction coefficients at layer.
        Index = experimental time, Columns = Line
        Smooth over time (window) by moving average (ma) or meadian (meadian)
        """
        ch_extco_df = self.all_extco_df.xs(channel, level=0, axis=1).xs(layer, level=1, axis=1)
        ma_ch_extco_df = ch_extco_df.rolling(window=window, closed='right').mean().shift(-int(window/2)+1)
        # ma_ch_extco_df = ch_extco_df.iloc[::-1].rolling(window=window, closed='left').mean().iloc[::-1]

        return ma_ch_extco_df

    def get_ledparams_at_line(self, channel: int, line: int, param='sum_col_val', yaxis='led_id', window=1, n_ref=10) -> pd.DataFrame:
        """
        Get a Dataframe containing extinction coefficients at line.
        Index = experimental time, Columns = Layer
        Smooth over time (window) by moving average (ma) or meadian (meadian).
        The parameters are normalized to the average of the first n_ref images.
        If n_ref is set to False the absolute values are returned.
        """
        if channel == 0:
            led_params = self.ch0_ledparams_df
        elif channel == 1:
            led_params = self.ch1_ledparams_df
        elif channel == 2:
            led_params = self.ch2_ledparams_df
        index = 'height' if yaxis == 'height' else 'led_id'
        led_params = led_params.reset_index().set_index(['Experiment_Time[s]',index])
        ii = led_params[led_params['line'] == line][[param]]
        if n_ref == False:
            rel_i = ii
        else:
            i0 = ii.groupby([index]).agg(lambda g: g.iloc[0:n_ref].mean())
            rel_i = ii/i0

        rel_i = rel_i.reset_index().pivot(columns=index,index='Experiment_Time[s]')
        rel_i.columns  = rel_i.columns.droplevel()
        # rel_i_ma = rel_i.iloc[::-1].rolling(window=window, closed='left').mean().iloc[::-1]
        rel_i_ma = rel_i.rolling(window=window, closed='right').mean().shift(-int(window/2)+1)

        return rel_i_ma