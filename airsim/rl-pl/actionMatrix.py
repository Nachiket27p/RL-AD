import numpy as np
import math

cols = 17
rows = 9
origin = (rows * cols) - math.ceil(float(cols)/2)
arr = [[None for i in range(cols)] for j in range(rows)]

gasbreak = 1.0  # negative means break. pos means gas
for i in range(rows):
    steer = -1.0
    for j in range(cols):
        arr[i][j] = (steer, gasbreak)
        steer += 2.0/(cols-1)
    gasbreak -= 1.0/(rows-1)

# print(arr)

for i in range(rows * cols):
    x, y = arr[i // cols][i % cols]
    print(x, y)

