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

actions: list[int] = [1, 0, 0, 1, 1]

print(f'Timestep {0}: {prior}')

for i, action in enumerate(actions):
    posterior = prior @ T[action]
    posterior = posterior / np.sum(posterior)

    print(f'Timestep {i + 1}: {posterior}')

    prior = posterior

""" Result:
Timestep 0: [0.5 0.5]
Timestep 1: [0.1 0.9]
Timestep 2: [0.1 0.9]
Timestep 3: [0.1 0.9]
Timestep 4: [0.02 0.98]
Timestep 5: [0.004 0.996]
"""