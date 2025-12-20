from scipy.stats import beta 
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 1)


#initial count of heads
a = 2

#initial count of tails
b = 2

#support
lb, ub = beta.support(a,b)

def update_belief(a, b, outcome):
    #increments heads or tails 
    if outcome == 'H':
        a += 1
    elif outcome == 'T':
        b += 1
    return a, b

def plot_belief(a, b):
    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')

for i in range(10):
    a, b = update_belief(a, b, 'H')
    plot_belief(a, b)

plt.legend()
plt.savefig("bayesian_update.png", dpi=150, bbox_inches='tight')
print("Figure saved as bayesian_update.png")
plt.close()