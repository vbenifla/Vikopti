import numpy as np
from vikopti.problems import Beam
from vikopti import Config
from vikopti.logger import Logger


def test():

    # Initialize default problem and config
    pb = Beam()
    config = Config()

    # Init logger
    config.save_dir = r"test_logger"
    logger = Logger(pb, config)

    # Test logger object
    for k in range(3):

        # Init data to log
        n = 20
        x = np.random.rand(n, pb.n_var)
        obj = np.random.rand(n, pb.n_obj)
        const = np.random.rand(n, pb.n_con)

        # Log data
        logger.log_pop(x[:10], obj[:10], const[:10])
        logger.log_reject(x[10:])
        logger.log_gen([(k + 1) * n, 5, 5, 10, 2, 3])


if __name__ == "__main__":
    test()
