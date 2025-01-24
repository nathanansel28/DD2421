from typing import Union, Literal, List, Tuple

from scipy.stats import uniform, norm
import numpy as np
import matplotlib.pyplot as plt
import math 


def calculate_entropy(
    data: List[int]
): 
    mydict = {}
    for item in data: 
        if item in mydict: 
            mydict[item] += 1
        else:
            mydict[item] = 1

    if len(mydict) == 0: 
        return 0.0 
    
    entropy = 0
    n = len(data)
    for item in mydict: 
        p_i = mydict[item]/n
        entropy += -1 * p_i * math.log(p_i, 2)

    return entropy


def generate_samples(
    type: Literal['uniform', 'non_uniform'],
    n: int, 
    mean: float = 0, 
    std: float = 10
) -> List[int]: 
    assert type in ['uniform', 'non_uniform']

    if type == 'uniform': 
        return list(range(1,n+1))
    elif type == 'non_uniform': 
        random_samples = np.random.normal(mean, std, size=n)
        return [round(num) for num in random_samples]


def generate_random_variables(
    type: Literal['uniform', 'gaussian']
): 
    """Ignore this for now."""

    assert type in ['uniform', 'gaussian']

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(uniform.ppf(0.01),
                    uniform.ppf(0.99), 100)

    if type == 'uniform': 
        ax.plot(x, uniform.pdf(x),
            'r-', lw=5, alpha=0.6, label='uniform pdf')
        rv = uniform()
        ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
        r = uniform.rvs(size=1000)
        print(r)

    elif type == 'gaussian':
        ax.plot(x, norm.pdf(x),
            'r-', lw=5, alpha=0.6, label='norm pdf')
        rv = norm()
        ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
        vals = norm.ppf([0.001, 0.5, 0.999])
        np.allclose([0.001, 0.5, 0.999], norm.cdf(vals))
        r = norm.rvs(size=1000)

    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.set_xlim([x[0], x[-1]])
    ax.legend(loc='best', frameon=False)
    plt.show()

    return r