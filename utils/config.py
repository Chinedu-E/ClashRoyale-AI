import easyocr
import keyboard


INPUT_DIM = (208,160,1)

CONV_FILTERS = [128,64,64, 128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]

DENSE_SIZE = 1024
CONV_T_FILTERS = [128,64,64,1]
CONV_T_KERNEL_SIZES = [4, 4, 4, 4]
CONV_T_STRIDES = [2, 2, 2, 2]


Z_DIM = 80

BATCH_SIZE = 100
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5


GAME_MON = {'top': 30, 'left': 0, 'width': 480, 'height': 830}

READER = easyocr.Reader(['en'])

NORMALIZE = lambda img: img/255 

