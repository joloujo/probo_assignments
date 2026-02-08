import numpy as np

transition_matrix = np.array([
    [0.0, 0.5, 0.5],
    [0.0, 0.8, 0.2],
    [0.0, 0.3, 0.7],
])

prior = np.array([1, 0, 0])

measurement_matrix = np.array([
    [0.0, 0.5, 0.5],
    [0.0, 0.9, 0.1],
    [0.0, 0.1, 0.9],
])

observations = [1, 2, 2, 1, 2]

# A list of the forward step state estimates, unnormalized
forward: list[np.ndarray] = []

for i, observation in enumerate(observations):
    unnormalized = prior * measurement_matrix[:, observation]
    forward.append(unnormalized)
    prior = unnormalized @ transition_matrix

# Normalize the forward step state estimates
filtered = [
    unnormalized / np.sum(unnormalized)
    for unnormalized in forward
]

backward: list[np.ndarray] = []

prior = np.array([1, 1, 1])

for i, observation in enumerate(reversed(observations)):
    # print(i, observation)
    unnormalized = prior * measurement_matrix[:, observation] @ transition_matrix.T
    # print(unnormalized)
    backward.append(unnormalized)

    prior = unnormalized 


for thing in backward: print(thing)
# for i, (u, n) in enumerate(zip(forward, filtered)):
#     print(f'Forward Step for k={i+1}: {u}')
#     print(f'Filtered Estimate for k={i+1}: {n}')