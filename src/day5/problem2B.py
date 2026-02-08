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
    unnormalized = prior * M[:, observation]
    forward.append(unnormalized)
    prior = unnormalized @ T[action]

# Normalize the forward step state estimates
filtered = [
    unnormalized / np.sum(unnormalized)
    for unnormalized in forward
]

for i, state in enumerate(filtered): print(f'Filtered at timestep {i + 1}: {state}')

""" Result:
Filtered at timestep 1: [0.66666667 0.33333333]
Filtered at timestep 2: [0.23529412 0.76470588]
Filtered at timestep 3: [0.09302326 0.90697674]
Filtered at timestep 4: [0.17021277 0.82978723]
Filtered at timestep 5: [0.01161103 0.98838897]
"""