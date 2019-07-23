import numpy as np
from Constant import *

a=np.random.randint(0,10,(5,4))
np.save('only_for_test/testfile/tf.npy',a)