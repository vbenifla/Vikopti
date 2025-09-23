import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from vikopti.utils import norm1D
from vikopti.problems import F1
from vikopti.fitness import compute_fitness, scale_fitness


def test():

    # Initialize problem
    pb = F1(True)

    # Design space
    n = 1000
    X = np.linspace(pb.bounds[0][0], pb.bounds[1][0], n, endpoint=True)
    x = X.reshape((n, pb.n_var))

    # Compute objective/constraints values
    res = np.array([pb.func(xi) for xi in x])
    obj = res[:, : pb.n_obj]
    const = res[:, pb.n_obj :]

    # Add some nans
    _obj = obj.copy()
    _obj[: int(n / 10), 0] = np.nan

    # Compute fitness and scaled fitness
    f, pen = compute_fitness(_obj, const)

    # cale fitness
    dist = cdist(x, x, "euclidean") / pb.sf
    fs, optima = scale_fitness(f, pen, dist)

    # Create a figure
    fig, ax = plt.subplots()
    ax.plot(X, norm1D(obj[:, 0]), label="obj")
    ax.plot(X, f, label="f", linestyle="--")
    ax.plot(X, fs, label="fs", linestyle="--")
    ax.legend()
    plt.show()


# def test2D():

#     # Init problem
#     pb = GLC()

#     # Design space
#     n = 100
#     x = np.linspace(pb.bounds[0][0], pb.bounds[1][0], n, endpoint=True)
#     y = np.linspace(pb.bounds[0][1], pb.bounds[1][1], n, endpoint=True)
#     X, Y = np.meshgrid(x, y)

#     # Flatten the grid to get all (x, y) points
#     xy_points = np.column_stack([X.ravel(), Y.ravel()])

#     # Compute objective/constraints values
#     res = np.array([pb.func(xy) for xy in xy_points])
#     obj = res[:, 0]
#     const = res[:, 1:]
#     Z = obj.reshape(X.shape)

#     # Compute fitness and scaled fitness
#     f, is_feasible, pen = compute_fitness(obj, const, alpha=1)
#     diff = xy_points[:, None, :] - xy_points[None, :, :]
#     dist = np.linalg.norm(diff, axis=-1) / pb.sf
#     d_mins = 0.1
#     n_mins = 10
#     fs, mins = scale_fitness(f, dist, is_feasible, d=d_mins, n=n_mins)
#     F = f.reshape(X.shape)
#     Fs = fs.reshape(X.shape)

#     # Plot Objective contour
#     fig, ax = plt.subplots()
#     cs = ax.contourf(X, Y, Z, levels=100, cmap=CMAP)
#     fig.colorbar(cs, ax=ax, label="Objective")
#     ax.scatter(xy_points[mins, 0], xy_points[mins, 1], color="cyan", s=50, edgecolors="black")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_title("Objective function contour")

#     # Plot Fitness contour
#     fig, ax = plt.subplots()
#     cs = ax.contourf(X, Y, F, levels=100, cmap=CMAP)
#     fig.colorbar(cs, ax=ax, label="Fitness")
#     ax.scatter(xy_points[mins, 0], xy_points[mins, 1], color="cyan", s=50, edgecolors="black")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_title("Fitness contour")

#     # Plot Scaled Fitness contour
#     fig, ax = plt.subplots()
#     cs = ax.contourf(X, Y, Fs, levels=100, cmap=CMAP)
#     fig.colorbar(cs, ax=ax, label="Scaled Fitness")
#     ax.scatter(xy_points[mins, 0], xy_points[mins, 1], color="cyan", s=50, edgecolors="black")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_title("Scaled Fitness contour")

#     plt.show()
if __name__ == "__main__":
    test()
