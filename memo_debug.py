import numpy as np

x= np.ones(100)
num = int(np.log2(len(x)))
print(np.exp2(num+1))
print(x)

while(len(x)<np.exp2(num+1)):
    x = np.append(x,0)

index = np.arange(len(x))

print(len(x))
print(num)
print(index)