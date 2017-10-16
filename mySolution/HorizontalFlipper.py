import glob
import os

from mySolution.VD_functions import *

notcars = glob.glob('../data_set_ext/non-vehicles/**/*.png', recursive=True)
cars = glob.glob('../data_set_ext/vehicles/**/*.png', recursive=True)
print("Number of car images:", len(cars))
print("Number of background images:", len(notcars))
print("Total number of samples is:", len(cars) + len(notcars))

for car in cars:
    path, filename = os.path.split(car)
    image = cv2.imread(car)
    flipped = cv2.flip(image, flipCode=1)
    cv2.imwrite(path+'/f_'+filename, flipped)

for notcar in notcars:
    path, filename = os.path.split(notcar)
    image = cv2.imread(notcar)
    flipped = cv2.flip(image, flipCode=1)
    cv2.imwrite(path+'/f_'+filename, flipped)

