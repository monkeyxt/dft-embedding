# pylint: disable=invalid-name
"""CP2K + Qiskit Nature embedding.

Usage:
    python dft-emb-client.py
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
from qiskit_nature.logging import logging as nature_logging
from qiskit_nature_cp2k.cp2k_integration import CP2KIntegration
from qiskit_nature_pyscf import PySCFGroundStateSolver

from pyscf import fci

np.set_printoptions(linewidth=500, precision=6, suppress=True)

logger = logging.getLogger(__name__)

level = logging.DEBUG

nature_logging.set_levels_for_names(
    {
        __name__: level,
        "qiskit": level,
        "qiskit_nature": level,
        "qiskit_nature_pyscf": level,
        "qiskit_nature_cp2k": level,
    }
)


def log_ci_states(fci_solver, uhf, ovlpab=None, thresh=1e-6):
    norb = fci_solver.norb
    nel_a, nel_b = fci_solver.nelec
    occslst_a = fci.cistring.gen_occslst(range(norb), nel_a)
    occslst_b = fci.cistring.gen_occslst(range(norb), nel_b)

    for root in range(fci_solver.nroots):
        if fci_solver.nroots == 1:
            ci_vector = fci_solver.ci
        else:
            ci_vector = fci_solver.ci[root]

        logger.info(
            f"Logging CI vectors and coefficients > {thresh} for root number {root}:"
        )

        pad = 4 + norb
        logger.info(f'  {"Conf": <{pad}} CI coefficients')
        for i, occsa in enumerate(occslst_a):
            for j, occsb in enumerate(occslst_b):
                if abs(ci_vector[i, j]) < thresh:
                    continue
                # generate the CI string and log it
                occ = ""
                for k in range(norb):
                    if k in occsa and k in occsb:
                        occ += "2"
                    elif k in occsa and k not in occsb:
                        occ += "u"
                    elif k not in occsa and k in occsb:
                        occ += "d"
                    else:
                        occ += "0"
                logger.info("  %s     %+.8f" % (occ, ci_vector[i, j]))


if __name__ == "__main__":
    HOST = "embedding_socket"
    PORT = 12345
    UNIX = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--uhf", action="store_true")
    parser.add_argument("--nroots", type=int, default=1)
    parser.add_argument("--spin", type=int, default=None)
    args = parser.parse_args()

    if args.uhf:
        fci_solver = fci.direct_uhf.FCI()
    else:
        fci_solver = fci.direct_spin1.FCI()
    fci_solver.nroots = args.nroots
    if args.spin is not None:
        fci_solver.spin = args.spin
    algo = PySCFGroundStateSolver(fci_solver)

    integ = CP2KIntegration(algo)
    integ.connect_to_socket(HOST, PORT, UNIX)
    integ.run()
    log_ci_states(integ.algo.solver, args.uhf, ovlpab=integ._overlap_ab, thresh=1e-3)
