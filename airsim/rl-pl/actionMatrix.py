import numpy as np
import math

cols = 17
rows = 9
# selects the middle element in the last row
origin = (rows * cols) - math.ceil(float(cols) / 2)
actionMatrix = [[None for i in range(cols)] for j in range(rows)]

gasbreak = 1.0  # pos means gas, zero means break
for i in range(rows):
    steer = -0.5
    for j in range(cols):
        actionMatrix[i][j] = (steer, gasbreak)
        steer += 1.0 / (cols - 1)
    gasbreak -= 1.0 / (rows - 1)

for row in actionMatrix:
    print(row)


