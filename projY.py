import numpy as np
import mmh3
seedValueY = np.random.randint(0, 2147483647)
seedSignY  = np.random.randint(0, 2147483647)
dims = []
sign = []
ydim = int(input())
projYdim = int(input())
for i in range(ydim):
	z = (mmh3.hash('azv'+str(i),seedValueY,signed = False))
	dims.append(z%projYdim)
	y = mmh3.hash('azv'+str(i),seedSignY,signed = False)
	sign.append((y%2)*2-1)

for p in dims:
	print(p)
for s in sign:
	print(s)