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

# a = np.array([[150, 160, 170, 180, 190], 
#               [55, 60, 65, 60, 70]])
# print(np.cov(a))

# spectrogram = np.abs(np.random.uniform(low=0, high=1, size=(5,6)))

# Y = np.array(spectrogram.T) # f * t
# f,t = np.shape(Y)
# k = 4
# # H, U を作成 Y = H * U
# np.random.seed(91)
# H = np.abs(np.random.uniform(low=0, high=1, size=(f,k)))
# U = np.abs(np.random.uniform(low=0, high=1, size=(k,t)))

# print(H.T[0])

# NMF = H @ U
# print(np.mean(Y-NMF))

# # 更新式
# for i in range(100):
#     H = H * (Y @ U.T) / (U @ (H @ U).T).T
#     U = U * (Y.T @ H).T / (H.T @ (H @ U))

# NMF = H @ U
# print(np.mean(Y-NMF))

# for i in range(k):
#     HH = np.zeros_like(H)
#     HH.T[i] = H.T[i]
#     print(HH)
#     print(HH @ U)

# x = 1
# def f (x):
#     x=x+1
#     print(x)
# f(x)
# print(x)

a = np.arange(6).reshape(2, 3)

np.savetxt('data/test1.txt', a)

b = np.loadtxt('data/test1.txt')
print(b)