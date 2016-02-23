__author__ = 'alex'
import math

def f1(x, n):
    x = float(x)
    if n - x < 3:
        return math.pow((.1 / math.sqrt(x)) + (.9 / (math.sqrt(x) + n - x)), -1)
    else:
        return math.pow((.1 / math.sqrt(x)) + (.5 / (math.sqrt(x) + n - x)) + .4 / (math.sqrt(x) + 3), -1)

n = 64
for i in range(1, n+1):
    print i, f1(i, n)