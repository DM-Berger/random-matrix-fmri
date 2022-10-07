import numpy as np

# fmt: off
TRIM_ARG_DEFAULTS     = "(1,:)"
UNFOLD_ARG_DEFAULTS   = {"smoother": "poly", "degree": 7, "detrend": False}
LEVELVAR_ARG_DEFAULTS = {"L": np.arange(0.5, 10, 0.1), "tol": 0.01, "max_L_iters": 50000}
RIGIDITY_ARG_DEFAULTS = {"L": np.arange(2, 20, 0.5)}
BRODY_ARG_DEFAULTS    = {"method": "mle"}
# fmt: on
