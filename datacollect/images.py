import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rawpy

class ImageData:
    def __init__(self, path_images: str, path_simulation: str):
        self.path_images = path_images
        self.path_simulation = path_simulation
        self.image_info_df = pd.read_csv(os.path.join(self.path_simulation, 'analysis/image_infos_analysis.csv'))
        self.led_info_path = os.path.join(self.path_simulation, 'analysis/led_search_areas_with_coordinates.csv')

    def get_image_name_from_timestep(self, timestep: int):
        """Returns imagename of image according to a certain timestep"""
        imagename = self.image_info_df.loc[self.image_info_df['Experiment_Time[s]'] == timestep]['Name'].values[0]
        return imagename

    def get_pixel_cordinates_of_LED(self, led_id: int):
        """Returns list with x and y pixel coordinates of LED"""
        led_info_df = pd.read_csv(self.led_info_path)
        pixel_positions = led_info_df.loc[led_info_df.index == led_id][[' pixel position x', ' pixel position y' ]].values[0]
        return pixel_positions

    def read_file(self, filename: str, channel: int, colordepth=14):
        """
        Returns a 2D array of channel values depending on the colordepth.
        14bit is default range for RAW. Bayer array is a 2D array where
        all channel values except the selected channel are set to zeroes.
        """
        with rawpy.imread(filename) as raw:
            data = raw.raw_image_visible.copy()
            filter_array = raw.raw_colors_visible
            black_level = raw.black_level_per_channel[channel]
            white_level = raw.white_level
        channel_range = 2 ** colordepth - 1
        channel_array = data.astype(np.int16) - black_level
        channel_array = (channel_array * (channel_range / (white_level - black_level))).astype(np.int16)
        channel_array = np.clip(channel_array, 0, channel_range)
        if channel == 0 or channel == 2:
            channel_array = np.where(filter_array == channel, channel_array, 0)
        elif channel == 1:
            channel_array = np.where((filter_array == 1) | (filter_array == 3), channel_array, 0)
        return channel_array

    def get_led_array(self, led_id: int, timestep: int, channel: int, radius: int):
        """
        Return a cropped 2D array with shape (2*radius, 2*radius)
        of channel values from an image according to a certain timestep.
        """
        imagename = self.get_image_name_from_timestep(timestep)
        path_image = os.path.join(self.path_images, imagename)
        pixel_position = self.get_pixel_cordinates_of_LED(led_id)
        channel_array = self.read_file(path_image, channel, radius)
        x = pixel_position[0]
        y = pixel_position[1]
        channel_array_cropped = channel_array[x - radius:x + radius, y - radius:y + radius]
        return channel_array_cropped
