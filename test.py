from calendar import c
import numpy as np
import matplotlib.pyplot as plt
import platypus as pl
import vis


if __name__ == "__main__":
    # Initialise a problem.
    problem = pl.DTLZ2(5)
    optimiser = pl.NSGAIII(problem, 12)
    optimiser.run(1000)

    # Produce a parallel coordinate plot.
    X, Y = vis.unpack_platypus(optimiser)
    #r = vis.rank_best_obj(Y)
    #vis.parallel_coords(Y, ["rbgcmyk"[c] for c in r])

    # Produce a PCA projection.
    r = vis.average_rank(Y) / Y.shape[0]
    vis.pca_projection(Y, colours=r)

    # Show the visualisations.
    plt.show()