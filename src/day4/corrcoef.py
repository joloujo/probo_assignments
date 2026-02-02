import numpy as np
import numpy.typing as nt
from cov import cov

def corrcoef(m: nt.NDArray) -> nt.NDArray:
    """
    
    
    Args:
        m (ArrayLike): The input matrix, where each row is a variable and each column is an observation
    """

    # Get the covariance matrix
    covariance = cov(m)

    # print(f'{covariance=}')

    # Make a matrix from the diagonal. Gives a matrix where X[i][j] = sqrt(C[i][i]C[j][j])
    denom_matrix = np.sqrt(np.outer(covariance.diagonal(), covariance.diagonal()))

    # print(f'{denom_matrix=}')

    correlation_coefficient_matrix = covariance / denom_matrix

    return correlation_coefficient_matrix
    

# Testing
def print_tests(m: nt.NDArray, name: str = 'm'):
    print(f'Testing {name}:')
    print(f'{name}:\n{m}\n')
    print(f'np.corrcoef({name}):\n{np.corrcoef(m)}\n')
    print(f'corrcoef({name}):\n{corrcoef(m)}\n')

if __name__ == '__main__':
    rng = np.random.default_rng(seed=42)
    xarr = rng.random((3, 3))
    print_tests(xarr, 'xarr')
