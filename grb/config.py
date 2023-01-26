import logging
import os

ROOT_PATH = 'C:/Python projects/GRB_data_processing/'
IMAGE_PATH = f'{ROOT_PATH}pics/'
ACS_DATA_PATH = 'E:/ACS'

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', filename=f'{ROOT_PATH}logs/api.log',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')