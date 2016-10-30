import importlib
import numpy as np

lib = importlib.import_module("1743734-lib")

a, b = 2.0, 4.0

x, y = lib.gendata(np.array([b, a]), 200)
print(lib.descent(y, x, 0.0001, 100))
lib.genplot(x, y, a=a, b=b)

