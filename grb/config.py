import logging

ROOT_PATH = 'C:/Python projects/GRB_data_processing/'
IMAGE_PATH = f'{ROOT_PATH}pics/'
LIGHT_CURVE_SAVE = f'{ROOT_PATH}light_curves/'
ACS_DATA_PATH = 'E:/ACS/'
GBM_DETECTOR_CODES = {0:'n0',1:'n1',2:'n2',3:'n3',4:'n4',5:'n5',6:'n6',7:'n7',8:'n8',9:'n9',10:'na',11:'nb',12:'b0',13:'b1'}

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', filename=f'{ROOT_PATH}logs/api.log',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')