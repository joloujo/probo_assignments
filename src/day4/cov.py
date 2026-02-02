import numpy as np
import numpy.typing as nt

def cov(m: nt.NDArray) -> nt.NDArray:
    """
    Find the covariance matrix for a set of variables
    
    Args:
        m (ArrayLike): The input matrix, where each row is a variable and each column is an observation
    """

    # Find the means along the columns
    mean = np.mean(m, axis=1)

    # Center the data on the means
    centered = (m.T - mean).T

    # Get the un-normalized variance
    variance = centered @ centered.T

    # Normalize by the degrees of freedom
    normalized = variance / (m.shape[1] - 1)

    return normalized

# Testing
def print_tests(m: nt.NDArray, name: str = 'm'):
    print(f'Testing {name}:')
    print(f'{name}:\n{m}\n')
    print(f'np.cov({name}):\n{np.cov(m)}\n')
    print(f'cov({name}):\n{cov(m)}\n')

x = np.array([[0, 2], [1, 1], [2, 0]]).T
print_tests(x, 'x')

x = [-2.1, -1,  4.3]
y = [3,  1.1,  0.12]
X = np.stack((x, y), axis=0)
print_tests(X, 'X')