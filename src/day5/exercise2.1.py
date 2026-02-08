import numpy as np

transition_matrix = np.array([
    [0.0, 0.5, 0.5],
    [0.0, 0.8, 0.2],
    [0.0, 0.3, 0.7],
])

prior = np.array([1, 0, 0])

print(f'Timestep {0}: {prior = }')

for i in range(26):
    posterior = prior @ transition_matrix

    print(f'Timestep {i + 1}: {posterior = }')

    prior = posterior