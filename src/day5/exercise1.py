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


""" Result:
Timestep 0: prior = array([0.985, 0.015])
Timestep 1: posterior = array([0.94258373, 0.05741627])
Timestep 2: posterior = array([0.80408163, 0.19591837])
Timestep 3: posterior = array([0.50642674, 0.49357326])
Timestep 4: posterior = array([0.20414508, 0.79585492])
Timestep 5: posterior = array([0.06026308, 0.93973692])
Timestep 6: posterior = array([0.01577893, 0.98422107])
Timestep 7: posterior = array([0.00399198, 0.99600802])
Timestep 8: posterior = array([0.00100099, 0.99899901])
Timestep 9: posterior = array([2.50435720e-04, 9.99749564e-01])
Timestep 10: posterior = array([6.26206918e-05, 9.99937379e-01])
"""