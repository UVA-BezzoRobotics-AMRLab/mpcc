import sys
import yaml
import argparse
import numpy as np
import cvxpy as cp

from cvxpygen import cpg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="LP_codegen")
    parser.add_argument("--yaml", type=str, default="")
    args = parser.parse_args()

    yaml_file = args.yaml

    params = None
    if yaml_file != "":
        with open(yaml_file) as stream:
            try:
                params = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print("ERROR:", e, file=sys.stderr)
                exit(1)
    else:
        print(
            "ERROR: YAML file must be provided in order to generate LP code!",
            file=sys.stderr,
        )
        exit(1)

    n = params["tube_poly_degree"] + 1
    N = params["tube_num_samples"]

    # arc length domain
    domain = cp.Parameter(2, name="Domain")

    x = cp.Variable(n)

    # cost is maximizing area of the curve on domain
    coeffs = 1 / np.arange(1, n + 1)
    cost = coeffs @ x * (domain[1] - domain[0])

    A = cp.Parameter((2 * N, n), name="A_mat")
    b = cp.Parameter(2 * N, name="b_vec")

    problem = cp.Problem(cp.Minimize(-cost), [A @ x <= b])

    cpg.generate_code(problem, code_dir=args.dir, solver=cp.ECOS)


if __name__ == "__main__":
    main()
