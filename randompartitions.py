from itertools import count, izip
import matplotlib.pyplot as plt
from numpy import arange, array, exp, pi, sqrt
from numpy.random import RandomState

def conjugatePartition(p):
    """
    Find the conjugate of a partition.
    E.g. len(p) = max(conjugate(p)) and vice versa.
    D. Eppstein, August 2005.

    """
    p = sorted(filter(lambda x: x > 0, p), reverse=True)
    result = []
    j = len(p)
    if j <= 0:
        return result
    while True:
        result.append(j)
        while len(result) >= p[j-1]:
            j -= 1
            if j == 0:
                return result

# Slower, even though it doesn't need sorting
def conjugatePartition_slow(p):
    p = array(p[:], dtype=int)
    result = []
    while True:
        k = len(p)
        if k > 0:
            result.append(k)
        else:
            return result
        p -= 1
        p = p[p > 0]

def plot_YoungDiagram(partition, block_size=1, offset=(0, 0), fig_scale=0.3, margin_scale=0.2, notation='french', **kwargs):
    partition_width = max(partition)
    partition_height = len(partition)

    fig_width = fig_scale*block_size*(partition_width + 2*margin_scale)
    fig_height = fig_scale*block_size*(partition_height + 2*margin_scale)

    xlims = (offset[0] - margin_scale*block_size, offset[0] + block_size*(partition_width + margin_scale))
    if notation == 'french':
        sign = 1
        ylims = (offset[1] - margin_scale*block_size,
                 offset[1] + block_size*(partition_height + margin_scale))
    elif notation == 'english':
        sign = -1
        ylims = (offset[1] - block_size*(partition_height + margin_scale),
                 offset[1] + margin_scale)
    else:
        raise Exception('Unknown notation style. Choose between french and english notation.')

    plt.figure(figsize=(fig_width, fig_height))
    for level, part_size in zip(range(0, len(partition)), partition):
        # Plot part contour
        xs = [offset[0],
              offset[0] + part_size*block_size,
              offset[0] + part_size*block_size,
              offset[0],
              offset[0]]

        ys = [offset[1] + block_size*sign*level,
              offset[1] + block_size*sign*level,
              offset[1] + block_size*sign*(level+1),
              offset[1] + block_size*sign*(level+1),
              offset[1] + block_size*sign*level]

        plt.plot(xs, ys, linewidth=2, color='black', **kwargs)

        # Plot part interior
        for x in xrange(1, part_size):
            plt.plot([offset[0] + block_size*x, offset[0] + block_size*x],
                     [offset[1] + block_size*sign*level, offset[1] + block_size*sign*(level + 1)],
                     linewidth=2, color='black', **kwargs)

    #plt.title(tuple(partition))
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.grid('off')
    plt.axis('off')

def randomPartition_AD5(n, m=None, seed=None):
    '''Uniform sampling of partitions of n using PDC with deterministic second half
    Algorithm 5 in Arratia & DeSalvo 2011 arXiv:1110.3856v7 and DeSalvo's answer at
    http://stackoverflow.com/questions/2161406/how-do-i-generate-a-uniform-random-integer-partition
    
    Question: How do I generate a uniform random integer partition?
    Stephen DeSalvo answered Nov 7 '13 at 6:46

    The title of this post is a bit misleading. A random integer partition is by
    default unrestricted, meaning it can have as many parts of any size. The
    specific question asked is about partitions of n into m parts, which is a
    type of restricted integer partition.

    For generating unrestricted integer partitions, a very fast and simple
    algorithm is due to Fristedt, in a paper called The Structure of Random
    Partitions of Large Integer (1993). The algorithm is as follows:

    Set x = exp(-pi/sqrt(6n) ). Generate independent random variables Z(1),
    Z(2), ..., Z(n), where Z(i) is geometrically distributed with parameter
    1-x^i. IF sum i*Z(i) = n, where the sum is taken over all i=1,2,...,n, then
    STOP. ELSE, repeat 2. Once the algorithm stops, then Z(1) is the number of
    1s, Z(2) is the number of 2s, etc., in a partition chosen uniformly at
    random. The probability of accepting a randomly chosen set of Z's is
    asymptotically 1/(94n^3)^(1/4), which means one would expect to run this
    algorithm O(n^(3/4)) times before accepting a single sample.

    The reason I took the time to explain this algorithm is because it applies
    directly to the problem of generating a partition of n into exactly m parts.
    First, observe that

    The number of partitions of n into exactly m parts is equal to the number of
    partitions of n with largest part equal to m.

    Then we may apply Fristedt's algorithm directly, but instead of generating
    Z(1), Z(2), ..., Z(n), we can generate Z(1), Z(2), ..., Z(m-1), Z(m)+1 (the
    +1 here ensures that the largest part is exactly m, and 1+Z(m) is equal in
    distribution to Z(m) conditional on Z(m)>=1) and set all other Z(m+1),
    Z(m+2), ... equal to 0. Then once we obtain the target sum in step 3 we are
    also guaranteed to have an unbiased sample. To obtain a partition of n into
    exactly m parts simply take the conjugate of the partition generated.

    The advantage this has over the recursive method of Nijenhuis and Wilf is
    that there is no memory requirements other than to store the random
    variables Z(1), Z(2), etc. Also, the value of x can be anything between 0
    and 1 and this algorithm is still unbiased! Choosing a good value of x,
    however, can make the algorithm much faster, though the choice in Step 1 is
    nearly optimal for unrestricted integer partitions.

    If n is really huge and Fristedt's algorithm takes too long (and table
    methods are out of the question), then there are other options, but they are
    a little more complicated; see my thesis
    https://sites.google.com/site/stephendesalvo/home/papers for more info on
    probabilistic divide-and-conquer and its applications.

    '''

    if seed == None:
        rng = RandomState()
    elif isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    if n < 1:
        raise ValueError('n should be greater or equal to 1')
    if m and not 1 <= m <= n:
        raise ValueError('Number of parts m should satisfy 1 <= m <= n')
    if n == 1:
        return [1]
    if m == 1:
        return [n]
    
    idx_range = arange(2, n + 1, dtype=int)
    for x in count(start=1):
        Z = rng.geometric(1.0 - exp(-idx_range*pi/sqrt(6*n)), size=n-1) - 1
        if m:
            Z[m-1:] = 0
            Z[m-2] += 1

        k = n - (idx_range*Z).sum()
        if k >= 0 and rng.uniform() < exp(-k*pi/sqrt(6*n)):
            Z = [k] + list(Z)
            partition = [i for i, zi in izip(xrange(1, n+1), Z) for _ in xrange(zi)][::-1]
            return partition if not m else conjugatePartition(partition)
