import os
import csv
import yaml
from pathlib import Path


class Logger:
    def __init__(self, pb, config):
        """
        Logger object to log all VIKGA run info and results.

        Parameters
        ----------
        pb : Problem
            Algorithm's optimization problem.
        config : Config
            Algorithm's configuration.
        """
        # Set directory
        if config.save_dir is None:
            save_dir = Path.cwd() / "results" / pb.name / "run"
        else:
            save_dir = Path(config.save_dir)
            save_dir = Path.cwd() / "results" / pb.name / save_dir

        # Make directory
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        config.save_dir = str(save_dir)

        # Set log files paths
        self.pop_file = os.path.join(self.save_dir, "pop.csv")
        self.reject_file = os.path.join(self.save_dir, "reject.csv")
        self.gen_file = os.path.join(self.save_dir, "gen.csv")

        # Map files to their headers
        files_headers = {
            self.pop_file: pb.vars + pb.objs + pb.consts,
            self.reject_file: pb.vars,
            self.gen_file: [
                "c_pop",
                "c_cross",
                "c_mute",
                "c_reject",
                "c_mins",
                "min_idx",
            ],
        }

        # Initialize all log files
        for key, value in files_headers.items():
            with open(key, "w", newline="") as file:
                csv.writer(file).writerow(value)

        # Make config and problem files
        with open(os.path.join(self.save_dir, "config.yml"), "w") as file:
            yaml.dump(config.__dict__.copy(), file, default_flow_style=False)
        with open(os.path.join(self.save_dir, "pb.yml"), "w") as file:
            yaml.dump(
                {
                    "name": pb.name,
                    "vars": pb.vars,
                    "bounds": pb.bounds,
                    "objs": pb.objs,
                    "consts": pb.consts,
                    "limits": pb.limits,
                    "sf": pb.sf,
                },
                file,
                default_flow_style=False,
            )

    def log_pop(self, x, obj, const):
        """
        Log population design variables, objective and constraints values.

        Parameters
        ----------
        x : np.ndarray
            Design variables.
        obj : np.ndarray
            Objective values.
        const : np.ndarray
            Constraints values.
        """
        with open(self.pop_file, "a", newline="") as file:
            writer = csv.writer(file)
            for i in range(len(x)):
                row = list(x[i]) + list(obj[i]) + list(const[i])
                writer.writerow(row)

    def log_reject(self, x):
        """
        Log rejected design variables.

        Parameters
        ----------
        x : np.ndarray
            Design variables.
        """
        with open(self.reject_file, "a", newline="") as file:
            writer = csv.writer(file)
            for i in range(len(x)):
                row = list(x[i])
                writer.writerow(row)

    def log_gen(self, data):
        """
        Log generation data.

        Parameters
        ----------
        data : list
            List of data to log.
        """
        with open(self.gen_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
