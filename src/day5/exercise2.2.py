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

prior = np.array([1, 1, 1])
backward: list[np.ndarray] = [prior]

for i, observation in enumerate(reversed(observations)):
    # print(i, observation)
    unnormalized = prior * measurement_matrix[:, observation] @ transition_matrix.T
    # print(unnormalized)
    backward.append(unnormalized)

    prior = unnormalized 

backward.pop(-1)

smoothed = [
    (f * b) / np.sum(f * b)
    for (f, b) in zip(forward, reversed(backward))
]

for i, state in enumerate(forward): print(f'Forward for timestep {i+1}: {state}')
for i, state in enumerate(filtered): print(f'Filtered for timestep {i+1}: {state}')
for i, state in enumerate(backward): print(f'Backward for timestep {i+1}: {state}')
for i, state in enumerate(smoothed): print(f'Smoothed for timestep {i+1}: {state}')


""" Result:
Forward for timestep 1: [0.5 0.  0. ]
Forward for timestep 2: [0.    0.025 0.225]
Forward for timestep 3: [0.      0.00875 0.14625]
Forward for timestep 4: [0.        0.0457875 0.0104125]
Forward for timestep 5: [0.         0.00397538 0.01480163]
Filtered for timestep 1: [1. 0. 0.]
Filtered for timestep 2: [0.  0.1 0.9]
Filtered for timestep 3: [0.         0.05645161 0.94354839]
Filtered for timestep 4: [0.        0.8147242 0.1852758]
Filtered for timestep 5: [0.         0.21171513 0.78828487]
Backward for timestep 1: [1 1 1]
Backward for timestep 2: [0.5  0.26 0.66]
Backward for timestep 3: [0.15   0.2004 0.1164]
Backward for timestep 4: [0.0624   0.036984 0.079344]
Backward for timestep 5: [0.037554   0.01724064 0.05109624]
Smoothed for timestep 1: [1. 0. 0.]
Smoothed for timestep 2: [0.         0.04924109 0.95075891]
Smoothed for timestep 3: [0.         0.09338552 0.90661448]
Smoothed for timestep 4: [0.         0.63400703 0.36599297]
Smoothed for timestep 5: [0.         0.21171513 0.78828487]
"""