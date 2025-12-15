import numpy as np
from math import sqrt
from ..problem import Problem


class Beam(Problem):
    """
    Welded beam design optimization problem.
    Objective: Minimize the cost of the welded beam.
    Constraints: 7 inequality constraints (shear, bending, geometric, etc.).
    Reference: https://www.mathworks.com/help/gads/multiobjective-optimization-welded-beam.html
    Variables:
        - h: Thickness of the weld (in)
        - l: Length of the weld (in)
        - t: Width of the beam (in)
        - b: Thickness of the beam (in)
    """

    def __init__(self):
        super().__init__(
            bounds=[[0.125, 0.1, 0.1, 0.125], [5, 10, 10, 5]],
            n_con=7,
            name="beam",
            vars=["h", "l", "t", "b"],
            consts=["shear", "bending", "g0", "g1", "g2", "def", "buckling"],
        )

    def func(self, x):

        # Get variables
        h, l, t, b = x

        # Constants
        P = 6000.0  # Load (lbf)
        L = 14.0  # Length (in)
        E = 30e6  # Young's modulus (psi)
        G = 12e6  # Shear modulus (psi)

        # Compute objective: Cost
        f0 = 1.10471 * (h**2) * l + 0.04811 * t * b * (14.0 + l)

        # Compute constraints
        # Shear stress constraint
        tau_p = P / (np.sqrt(2) * h * l)
        M = P * (L + l / 2)
        R = np.sqrt((l**2) / 4 + ((h + t) / 2) ** 2)
        J = 2 * np.sqrt(2) * h * l * (l**2 / 12 + ((h + t) / 2) ** 2)
        tau_pp = M * R / J
        tau = np.sqrt(tau_p**2 + tau_pp**2 + tau_p * tau_pp * l / R)
        g0 = tau - 13600

        # Bending stress constraint
        sigma = 6 * P * L / (b * t**2)
        g1 = sigma - 30000

        # Geometric constraints
        g2 = h - b
        g3 = 0.10471 * h**2 + 0.04811 * t * b * (14 + l) - 5.0
        g4 = 0.125 - h

        # Deflection constraint
        delta = 4 * P * L**3 / (E * b * t**3)
        g5 = delta - 0.25

        # Buckling load constraint
        Pc = (
            (4.013 * E * np.sqrt(t**2 * l**6 / 36))
            / (L**2)
            * (1 - t / (2 * L) * np.sqrt(E / (4 * G)))
        )
        g6 = P - Pc

        return (f0, g0, g1, g2, g3, g4, g5, g6)

    # def __init__(self):
    #     super().__init__(
    #         bounds=[[0.1, 0.1, 0.1, 0.1], [2, 10, 10, 2]],
    #         name="beam",
    #         vars=["h", "l", "t", "b"],
    #     )

    # def func(self, x):

    #     y = 1.10471*x[0]**2*x[1]+0.04811*x[2]*x[3]*(14.0+x[1])

    #     # parameters
    #     P = 6000; L = 14; E = 30e+6; G = 12e+6;
    #     t_max = 13600; s_max = 30000; d_max = 0.25;

    #     M = P*(L+x[1]/2)
    #     R = sqrt(0.25*(x[1]**2+(x[0]+x[2])**2))
    #     J = 2*(sqrt(2)*x[0]*x[1]*(x[1]**2/12+0.25*(x[0]+x[2])**2));
    #     P_c = (4.013*E/(6*L**2))*x[2]*x[3]**3*(1-0.25*x[2]*sqrt(E/G)/L);
    #     t1 = P/(sqrt(2)*x[0]*x[1]); t2 = M*R/J;
    #     t = sqrt(t1**2+t1*t2*x[1]/R+t2**2);
    #     s = 6*P*L/(x[3]*x[2]**2)
    #     d = 4*P*L**3/(E*x[3]*x[2]**3);
    #     # Constraints
    #     g1 = t-t_max; #done
    #     g2 = s-s_max; #done
    #     g3 = x[0]-x[3];
    #     g4 = 0.10471*x[0]**2+0.04811*x[2]*x[3]*(14.0+x[1])-5.0;
    #     g5 = 0.125-x[0];
    #     g6 = d-d_max;
    #     g7 = P-P_c; #done

    #     g=[g1,g2,g3,g4,g5,g6,g7]
    #     g_round=np.round(np.array(g),6)
    #     w1=100
    #     w2=100

    #     phi=sum(max(item,0) for item in g_round)
    #     viol=sum(float(num) > 0 for num in g_round)

    #     reward = (y + (w1*phi + w2*viol))

    #     return (reward,)
