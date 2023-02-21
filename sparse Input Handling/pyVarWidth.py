import numpy as np
from numpy.random import default_rng
from time import time_ns
from tqdm import tqdm

UPPER = 100_000
LOWER = 1_000
STEP = 1_000

REPEAT = 20

TESTS = np.arange(LOWER,UPPER+STEP,STEP)
res = np.zeros((2,len(TESTS),REPEAT))

with tqdm(total=len(TESTS)) as pbar:
    for k, WIDTH in enumerate(TESTS):
        pbar.update()
        weights = np.random.randint(-40,30,(WIDTH,256))
        inp = np.zeros(WIDTH,int)
        rng = default_rng()
        numbers = rng.choice(WIDTH, size=int(WIDTH*0.3), replace=False)
        inp[numbers] = 1

        for i in range(REPEAT):
            inp_arr = np.reshape(inp,(1,WIDTH))
            start = time_ns()
            t = np.matmul(inp_arr,weights,dtype=np.float64)
            end = time_ns()
            res[0,k,i] = end - start

        for i in range(REPEAT):
            start = time_ns()
            out = np.zeros(256)
            for l in range(WIDTH):
                if inp[l] != 0:
                    out += weights[l,:]
            end = time_ns()
            res[1,k,i] = end - start

np.save("pyVarWidth30",res)