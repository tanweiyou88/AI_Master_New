import cv2
import numpy as np

r = np.array([0, 1, 2, 3, 4, 5, 6])
g = np.array([4, 4, 4, 4, 4, 4, 4])
b = np.array([2, 2, 2, 2, 2, 2, 2])
# dc = cv2.min(cv2.min(r,g),b)
# o1 = cv2.min(r,g)
# print(o1)
# o2 = cv2.min(o1,b)
# print(o2)

M = 10
N = 2
p = 0.1
# searchidx = (-r).argsort()[:int(M*N*p)]
print("r:",r)
print("-r:",-r)
print("(-r).argsort():",(-r).argsort())
print("M*N*p:",M*N*p)
print("(-r).argsort()[:int(M*N*p):",(-r).argsort()[:int(M*N*p)])

searchidx = (-r).argsort()[:int(M*N*p)]
print("searchidx.shape", searchidx.shape)

c = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]],[[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
flatc = c.reshape(c.shape[0]*c.shape[1], 3)
print("c:",c)
print("flatc:",flatc)
print("flatc.take(searchidx, axis=0)", flatc.take(searchidx, axis=0))
A = np.mean(flatc.take(searchidx, axis=0),dtype=np.float64, axis=0)
print("A:", A)
K = np.max(A)
print("K:", K)


print("r - b:", r - b)
print("r - K:", r - K)