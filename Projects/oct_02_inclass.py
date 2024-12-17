"""
In-class coding practice for optimization
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# SECTION 1. Utility function with one consumption
# 1.1 define the utility function


# 1.2 generate the data

# 1.3. visualization




# SECTION 2. Utility function with two consumptions
# 2.1. define the function


# 2.3. visualization 
# Fix consumption of x1, plot the relationship between x2 and u


# SECTION 3., Plot the relationship between x1, x2  and TU

# 3.1. define the utility function

# 3.2. generate data


# 3.3. visualization (3-D)
def plot_utility_3d(x1, x2, u_level):
    fig = plt.figure(figsize=(12, 12))
    ax_3d = fig.add_subplot(1, 1, 1, projection='3d')
    ax_3d.countour3d(x1,x2,u_level,cmap=plt.cm.Blues)
    ax_3d.set_xlabel('x1')
    ax_3d.set_ylabel('x2')
    ax_3d.set_zlabel('u')
    ax_3d.set_title('Utility function')
    plt.show()


# SECTION 4. the flatten view of indifference curve
def u_fn2(x1, x2):
    return x1**0.5 + x2**0.5

x1 = np.linspace(0, 100, 100)
x2 = np.linspace(0, 100, 100).reshape((100,1))
u_level = u_fn2(x1, x2)

def plot_utility_flat(x1, x2, u_level):
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,1,1)
    contours=ax1.contourf(x1,x2.flatten(),u_level, cmap=plt.cm.Blues)
    fig.colorbar(contours)
    ax1.set_xlabel('consumption of x1')
    ax1.set_ylabel('consumption of x2')
    ax1.set_title('Indifference Map')
    plt.show()
    
plot_utility_flat(x1, x2, u_level)
# SECTION 5. Define and plot the budget constraint
def budget(x1, p1, p2, w):
    return w/p2 - p1*x1/p2

def plot_budget(ax, p1, p2, w):
    x1 = np.linspace(0, w/p1, 10)
    x2 = budget(x1, p1, p2, w)
    ax.plot(x1,x2,label='budget constraint', marker='o')
    ax.fill_between(x1, x2, alpha=0.5)
    ax.set_xlabel('consumption of x1')
    ax.set_ylabel('consumption of x2')
    ax.set_title('Budget Constraint')
    plt.ylim(0, w/p2)
    ax.legend()
    return ax

# plot the budget constraint
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1)
plot_budget(ax, 2, 4, 100)
plt.show()