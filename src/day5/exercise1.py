import numpy as np

# Our initial prior
prior = np.array([0.985, 0.015])

# Our likelihood, or the probability of our observation given the state
# Because our observation is always the same in this case, we don't have to use the whole matrix
likelihood = np.array([1/4, 1])

print(f'Timestep {0}: {prior = }')

for i in range(10):
    posterior = likelihood * prior
    posterior = posterior / np.sum(posterior)

    print(f'Timestep {i + 1}: {posterior = }')

    prior = posterior
