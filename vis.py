import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.stats as st


def unpack_platypus(optimiser):
    """
    Take the Pareto front approximation from a Platypus optimiser and
    return a tuple of Numpy arrays -- one holds the decision variables and
    the other hold the objective variables.
    """
    X = np.array([soln.variables for soln in optimiser.result])
    Y = np.array([soln.objectives for soln in optimiser.result])
    return X, Y


def rank_best_obj(Y):
    """
    Rank the given objective vectors according to the objective on which 
    they have the best rank.
    """
    N, M = Y.shape
    R = np.zeros((N, M))

    for m in range(M):
        R[:,m] = st.rankdata(Y[:,m])
    print(R)

    return R.argmin(axis=1).astype(np.int)


def parallel_coords(Y, colours=None, cmap="viridis", xlabels=None):
    """
    Produce a parallel coordinate plot for the objective space provided.
    """
    plt.figure()
    N, M = Y.shape

    if colours is None:
        colours = ["k"] * N         # Not really ideal, needs fixing.

    objTicks = np.arange(M, dtype=np.int)
    if xlabels is None:
        xlabels = objTicks + 1
    
    for i in range(N):
        plt.plot(objTicks, Y[i], c=colours[i])

    plt.xticks(objTicks, xlabels)
    plt.xlabel("Objective")
    plt.ylabel("$f(\mathbf{x})$")