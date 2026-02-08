import numpy as np

prior = np.array([0.5, 0.5])

T: dict[int, np.ndarray] = {
    0: np.array([
        [1, 0],
        [0, 1],
    ]),
    1: np.array([
        [0.2, 0.8],
        [0, 1],
    ]),
}

M: np.ndarray = np.array([
    [0.8, 0.2],
    [0.4, 0.6],
])

actions: list[int] = [1, 0, 0, 1, 1]
observations = [0, 0, 1, 0, 1]

forward: list[np.ndarray] = []

for i, (action, observation) in enumerate(zip(actions, observations)):
    unnormalized = prior @ T[action] * M[:, observation]
    forward.append(unnormalized)
    prior = unnormalized 

# Normalize the forward step state estimates
filtered = [
    unnormalized / np.sum(unnormalized)
    for unnormalized in forward
]

for i, state in enumerate(filtered): print(f'Filtered at timestep {i + 1}: {state}')

""" Result:
Filtered at timestep 1: [0.18181818 0.81818182]
Filtered at timestep 2: [0.30769231 0.69230769]
Filtered at timestep 3: [0.12903226 0.87096774]
Filtered at timestep 4: [0.05031447 0.94968553]
Filtered at timestep 5: [0.00337695 0.99662305]
"""