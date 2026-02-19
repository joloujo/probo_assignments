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

""" Result:
Timestep 0: prior = array([1, 0, 0])
Timestep 1: posterior = array([0. , 0.5, 0.5])
Timestep 2: posterior = array([0.  , 0.55, 0.45])
Timestep 3: posterior = array([0.   , 0.575, 0.425])
Timestep 4: posterior = array([0.    , 0.5875, 0.4125])
Timestep 5: posterior = array([0.     , 0.59375, 0.40625])
Timestep 6: posterior = array([0.      , 0.596875, 0.403125])
Timestep 7: posterior = array([0.       , 0.5984375, 0.4015625])
Timestep 8: posterior = array([0.        , 0.59921875, 0.40078125])
Timestep 9: posterior = array([0.        , 0.59960938, 0.40039062])
Timestep 10: posterior = array([0.        , 0.59980469, 0.40019531])
Timestep 11: posterior = array([0.        , 0.59990234, 0.40009766])
Timestep 12: posterior = array([0.        , 0.59995117, 0.40004883])
Timestep 13: posterior = array([0.        , 0.59997559, 0.40002441])
Timestep 14: posterior = array([0.        , 0.59998779, 0.40001221])
Timestep 15: posterior = array([0.       , 0.5999939, 0.4000061])
Timestep 16: posterior = array([0.        , 0.59999695, 0.40000305])
Timestep 17: posterior = array([0.        , 0.59999847, 0.40000153])
Timestep 18: posterior = array([0.        , 0.59999924, 0.40000076])
Timestep 19: posterior = array([0.        , 0.59999962, 0.40000038])
Timestep 20: posterior = array([0.        , 0.59999981, 0.40000019])
Timestep 21: posterior = array([0.       , 0.5999999, 0.4000001])
Timestep 22: posterior = array([0.        , 0.59999995, 0.40000005])
Timestep 23: posterior = array([0.        , 0.59999998, 0.40000002])
Timestep 24: posterior = array([0.        , 0.59999999, 0.40000001])
Timestep 25: posterior = array([0.        , 0.59999999, 0.40000001])
Timestep 26: posterior = array([0. , 0.6, 0.4])
"""

"""
The probability distribution converges
"""