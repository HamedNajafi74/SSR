import numpy as np
import matplotlib.pyplot as plt

def normalize_columns(matrix):
    max_vals = np.max(np.abs(matrix), axis=0)
    max_vals[max_vals == 0] = 1  # To avoid division by zero
    normalized_matrix = matrix / max_vals
    return normalized_matrix

def count_sign_changes(matrix):
    sign_changes = np.sum(np.diff(np.sign(matrix), axis=0) != 0, axis=0)
    return sign_changes

# User Input
"""
num_parts = int(input("Enter number of rotor parts: "))
fb = float(input("Enter the base frequency in Hz: "))
mass = np.array([float(input(f"Enter mass of part {i+1}: ")) for i in range(num_parts)])
#stiffness = np.array([float(input(f"Enter stiffness between parts {i+1} and {i+2}: ")) for i in range(num_parts-1)])
"""


#Problem 0
"""
fb = 50
num_parts = 6
mass = [0.129 , 0.2161, 1.1926, 1.2281, 1.2062, 0.0045] 
stiffness = [16.0858, 29.1075, 43.365,  59.0483,  2.3517]
"""


#Problem 1
"""
fb = 60
num_parts = 5
mass = 2*np.array([0.124 , 0.232, 1.155, 1.192, 0.855]) 
stiffness = [21.8, 48.4, 75.6,  62.3,  1.98]
"""

#Problem 2
"""
fb = 60
num_parts = 5
mass = 2*np.array([0.176 , 1.427, 1.428, 1.428, 0.869]) 
stiffness = [17.78, 27.66, 31.31,  37.25]
"""

#Problem 3
"""
fb = 60
num_parts = 4
mass = 2*np.array([0.099 , 0.337, 3.68, 0.946]) 
stiffness = [37.95, 81.91, 82.74]
"""

#Problem 4
fb = 60
num_parts = 6
mass = 2*np.array([0.254 , 0.983, 1.001, 1.009, 1.035, 0.013]) 
stiffness = [13.9, 18.2, 25.2,  54.9,  5.7]




# Torsional Stiffness Matrix
K = np.zeros((num_parts, num_parts))
for i in range(num_parts-1):
    K[i][i] += stiffness[i]
    K[i+1][i] -= stiffness[i]
    K[i][i+1] -= stiffness[i]
    K[i+1][i+1] += stiffness[i]

# Torsional Mass Matrix
M = np.zeros((num_parts, num_parts))
for i in range(num_parts):
    M[i][i] += mass[i]

# Eigenvalue Problem
A1=np.matmul(np.linalg.inv(M), K)

eigenvalues, eigenvectors = np.linalg.eig(A1)

wb=2*np.pi*fb
freqs = np.sqrt(abs(eigenvalues)*wb) / (2*np.pi)

normalized_V = normalize_columns(eigenvectors)
n=num_parts
Modenumber = count_sign_changes(normalized_V)

print("A1=\n",np.round(A1, decimals=2))
print("L=\n",np.round(eigenvalues, decimals=2))
print("Frequencies(Hz)=\n",np.round(freqs, decimals=2))
print("V=\n",np.round(eigenvectors, decimals=2))
print("normalized_V=\n",np.round(normalized_V, decimals=2))


# Create subplots for each column

fig, axes = plt.subplots(n, 1, figsize=(n*1.2, 7))
for i, ax in enumerate(axes):
    x_values = np.arange(1, n+1)
    line, = ax.plot(x_values,normalized_V[:, i], marker='o', linestyle='-', label= "Column{}, Mode{},f = {} Hz".format(i+1, Modenumber[i],np.round(freqs[i], decimals=2)))
    #ax.set_xlabel('Row')
    #ax.set_ylabel('Normalized Value')
    #ax.set_title('Column {}'.format(i+1))
    #ax.legend()
    #ax.grid(True)

    # Add vertical grid with dashed lines
    ax.grid(axis='x', linestyle='dashed')

     # Remove tick numbers on both axes
    ax.set_xticklabels([])
    #ax.set_yticklabels([])
    ax.set_yticks([-1,0,1])
    ax.set_ylim(-1.5, 1.5)

    # Remove the black box around each subplot
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)

    # Position the x-axis in the middle
    ax.spines['bottom'].set_position('center')
    ax.spines['bottom'].set_color('gray')

    # Create the legend at the right side
    legend_text = 'f ' + r'$_{{{}}}$'.format(Modenumber[i]) + ' = {} Hz'.format(np.round(freqs[i], decimals=2))
    #ax.legend([legend_text], loc='center left', bbox_to_anchor=(1, 0.5))

    # Add the legend_text as a text in the plot
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    x_text = x_range[1] + 0.05 * (x_range[1] - x_range[0])  # Adjust the position
    y_text = (y_range[0] + y_range[1]) / 2  # Middle of the y-axis
    ax.text(x_text, y_text, legend_text, ha='left', va='center')

    # Label each point with its y-axis value
    for x, y in zip(x_values, normalized_V[:, i]):
        ax.text(x, y, '{:.2f}'.format(y), ha='center', va='bottom')

plt.tight_layout()
plt.show()
