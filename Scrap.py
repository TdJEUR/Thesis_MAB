from Helpers import generate_combinations
import pandas as pd

size = 3
x = 0
y = 0.2
dt = 0.1

combinations = generate_combinations(size, x, y, dt)

print(combinations)


