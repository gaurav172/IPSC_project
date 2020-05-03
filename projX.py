import numpy as np
import mmh3
seedValueX = np.random.randint(0, 2147483647)
seedSignX  = np.random.randint(0, 2147483647)
dims = []
sign = []
xdim = int(input())
projXdim = int(input())
for i in range(xdim):
	z = (mmh3.hash('azv'+str(i),seedValueX,signed = False))
	dims.append(z%projXdim)
	y = mmh3.hash('azv'+str(i),seedSignX,signed = False)
	sign.append((y%2)*2-1)

for p in dims:
	print(p)
for s in sign:
	print(s)