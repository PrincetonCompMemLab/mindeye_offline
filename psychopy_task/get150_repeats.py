import random
import copy
import pandas as pd
import pdb
import os
import numpy as np

random.seed(1)
images_paths = os.listdir("images/")
random.shuffle(images_paths)
images150 = images_paths[:150]
print(images150)
import pickle
# open a file, where you ant to store the data
file = open('images150', 'wb')

# dump information to that file
pickle.dump(images150, file)

# close the file
file.close()