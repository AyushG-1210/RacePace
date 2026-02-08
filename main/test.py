import fastf1
session = fastf1.get_session(2025, 'Zandvoort', 'R')
session.load()
circuit_info = session.get_circuit_info()
x = circuit_info.corners
print(x)

import matplotlib.pyplot as plt
plt.plot(x["X"], x["Y"], marker='o')
plt.title("Zandvoort Circuit Corners")
plt.show()