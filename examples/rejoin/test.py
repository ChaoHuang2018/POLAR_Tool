import numpy as np
x = (np.random.random(4) - .5) * 3;
print(x);
print(np.clip(x[0], -.5, .5))
print(np.clip(x[2], -1., 1.))

num_outputs = 4
w = np.zeros([num_outputs, 4 + 1 + 4 + 1])
b = np.zeros([4 + 1 + 4 + 1])
w[0][0] = 1.
w[0][1] = -1.
w[0][2] = -1.
w[0][3] = 1.
w[1][4] = 1.
w[2][5] = 1.
w[2][6] = -1.
w[2][7] = -1.
w[2][8] = 1.
w[3][9] = 1.

b[0] = -0.5
b[1] = 0.5
b[2] = -0.5
b[3] = 0.5
b[5] = -1.0
b[6] = 1.0
b[7] = -1.0
b[8] = 1.0

x = np.matmul(w.T, x) + b 
x = x * (x >= 0.)
w = np.zeros([4 + 1 + 4 + 1, num_outputs])
b = np.zeros([num_outputs])

w[0][0] = 0.5
w[1][0] = -0.5
w[2][0] = 0.5
w[3][0] = 0.5
w[4][1] = 1.
w[5][2] = 0.5
w[6][2] = -0.5
w[7][2] = 0.5
w[8][2] = 0.5
w[9][3] = 1.


x = np.matmul(w.T, x) + b 


# min(x[0], 1)= - max(-x[0], -0.5)
# min(x[2], 1.0) = - max(-x[2], -1.0)
w = np.zeros([num_outputs, 4 + 1 + 4 + 1])
b = np.zeros([4 + 1 + 4 + 1])
w[0][0] = -1.
w[0][1] = 1.
w[0][2] = 1.
w[0][3] = -1.
w[1][4] = 1.
w[2][5] = -1.
w[2][6] = 1.
w[2][7] = 1.
w[2][8] = -1.
w[3][9] = 1.
b[0] = -0.5
b[1] = 0.5
b[2] = -0.5
b[3] = 0.5
b[5] = -1.0
b[6] = 1.0
b[7] = -1.0
b[8] = 1.0

x = np.matmul(w.T, x) + b 
x = x * (x >= 0.)

w = np.zeros([4 + 1 + 4 + 1, num_outputs])
b = np.zeros([num_outputs])
w[0][0] = -0.5
w[1][0] = 0.5
w[2][0] = -0.5
w[3][0] = -0.5
w[4][1] = 1.
w[5][2] = -0.5
w[6][2] = 0.5
w[7][2] = -0.5
w[8][2] = -0.5
w[9][3] = 1.

x = np.matmul(w.T, x) + b 
print(x)