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

prior = np.array([1, 1])
backward: list[np.ndarray] = [prior]

for i, (action, observation) in enumerate(zip(reversed(actions), reversed(observations))):
    unnormalized = prior * M[:, observation] @ T[action].T
    backward.append(unnormalized)

    prior = unnormalized 

backward.pop(-1)

smoothed = [
    (f * b) / np.sum(f * b)
    for (f, b) in zip(forward, reversed(backward))
]

# for i, state in enumerate(forward): print(f'Forward for timestep {i+1}: {state}')
# for i, state in enumerate(filtered): print(f'Filtered for timestep {i+1}: {state}')
for i, state in enumerate(backward): print(f'Backward for timestep {i+1}: {state}')
for i, state in enumerate(smoothed): print(f'Smoothed for timestep {i+1}: {state}')

""" Result:
Backward for timestep 1: [1 1]
Backward for timestep 2: [0.52 0.6 ]
Backward for timestep 3: [0.2752 0.24  ]
Backward for timestep 4: [0.05504 0.144  ]
Backward for timestep 5: [0.044032 0.0576  ]
Smoothed for timestep 1: [0.60456942 0.39543058]
Smoothed for timestep 2: [0.10523096 0.89476904]
Smoothed for timestep 3: [0.10523096 0.89476904]
Smoothed for timestep 4: [0.1509434 0.8490566]
Smoothed for timestep 5: [0.01161103 0.98838897]
"""