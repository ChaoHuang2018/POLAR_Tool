import matplotlib.pyplot as plt
import numpy as np

xys = []
f = open("bp_relu_plt.txt", "r")
line = f.readline()
while line:
    xys.append([float(i) for i in line.split("\n")[0].split(" ")])
  
    line = f.readline()
f.close()

xys = np.asarray(xys)
 
plt.title(label = "The interval remainder for the TM is [-BP(0)/2, BP(0)/2]")
xs = xys[:, 0]
ys1 = xys[:, 1]
ys2 = xys[:, 2]
ys3 = xys[:, 3]
ys4 = xys[:, 4]
xys = None
plt.plot(xs, ys1, label = "ReLU(x)")
plt.plot(xs, ys2, linestyle='dashed')
plt.plot(xs, ys3, label = "BP(x)-BP(0)/2")
plt.plot(xs, ys4, linestyle='dashed')

plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()

