from paddle_haptic_client import weight
import numpy as np
import matplotlib.pyplot as plt

r_list = np.linspace(0.0, 1.5, 100)
w_list = []

for r in r_list:
    w = weight(r, r_min=0.3, r_far=1.2, weight_max=100.0, delta=0.05)
    w_list.append(w)

plt.plot(r_list, w_list)
plt.xlabel('r')
plt.ylabel('w')
plt.title('Weight function')
plt.show()