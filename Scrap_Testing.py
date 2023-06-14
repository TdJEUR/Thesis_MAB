import numpy as np
import math
from scipy.stats import skew
from Helpers import calc_skew


l1 = [0.01, 0.06]
l2 = [1] * 5

print(calc_skew(l1))
print(calc_skew(l2))