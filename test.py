import numpy as np
import matplotlib.pyplot as plt

f = 900e6 
d0 = 1 
gamma = 3.2  
Gt = 0  
Gr = 0  

# required power 
Prx = -80


c = 3e8 
L_d0 = 20 * np.log10((4 * np.pi * d0 * f) / c)
distances = np.linspace(1, 10000, 100)

# Prx(d) = Prx(d0)(in db) + 10 * gamma * log10(d0/d)
path_loss = L_d0 + 10 * gamma * np.log10(distances / d0)


Ptx = Prx + path_loss - Gt - Gr

plt.figure(figsize=(8, 6))
plt.plot(distances, Ptx, label='Required Transmission Power', color='r')
plt.title('Required Transmission Power vs. Distance', fontsize=14)
plt.xlabel('Distance [m]', fontsize=12)
plt.ylabel('Transmission Power [dBm]', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

max_power = 20
index = np.abs(Ptx - max_power).argmin()
print("Max distance: ", distances[index])
