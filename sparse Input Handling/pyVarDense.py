import numpy as np
from time import time_ns
from tqdm import tqdm

weights = np.zeros((50_000,256),int)
for i in range(256):
    weights[:,i] = np.loadtxt(f"numberfiles/weights/W{i}.txt")

res = np.zeros((19990,100))

for l in tqdm(range(10,20_000)):
    inp = np.loadtxt(f"numberfiles/numb/N{l}.txt",int)
    for k in range(100):
        start = time_ns()
        np.matmul(inp,weights,dtype=np.float64)
        end = time_ns()
        diff = end - start
        res[l-10,k] = diff

np.save("pyVarResDense",res)