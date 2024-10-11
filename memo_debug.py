import numpy as np

# x= np.ones(100)
# num = int(np.log2(len(x)))
# print(np.exp2(num+1))
# print(x)

# while(len(x)<np.exp2(num+1)):
#     x = np.append(x,0)

# index = np.arange(len(x))

# print(len(x))
# print(num)
# print(index)

# x= np.arange(6).reshape(2,3)
# print(np.size(x))
# print(len(x))

# x=[1,2,3]
# x=x*10 +3
# print(x)

a = np.array([[150, 160, 170, 180, 190], 
              [55, 60, 65, 60, 70]])
print(np.cov(a))