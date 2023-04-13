import numpy as np
import matplotlib.pyplot as plt

b, a = np.meshgrid(np.linspace(0, 30, 60), np.linspace(0, 30, 60))

c =  0.2 + 2.0 * 0.001027 * np.sqrt( a ** 2 + b ** 2)
c = c[:-1, :-1]
l_a=a.min()
r_a=a.max()
l_b=b.min()
r_b=b.max()
l_c,r_c  = 0.2, np.abs(c).max()

figure, axes = plt.subplots()

c = axes.pcolormesh(a, b, c, cmap='copper', vmin=l_c, vmax=r_c)
 
axes.axis([l_a, r_a, l_b, r_b])
plt.xlabel('', fontsize=50) 
plt.ylabel('', fontsize=50) 
cbar = figure.colorbar(c)
#cbar.set_fontsize(18)

plt.show()

