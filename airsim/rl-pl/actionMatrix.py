import numpy as np

cols = 9
rows = 9
arr = [[None for i in range(cols)] for j in range(rows)]

gasbreak = -1.0  # negative means break. pos means gas
for i in range(rows):
    steer = -1.0
    for j in range(cols):
        arr[i][j] = (steer, gasbreak)
        steer += 0.25
    gasbreak += 0.25


x, y = arr[40 // 9][40 % 9]

print(x)
print(y)
