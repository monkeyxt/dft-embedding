# Steps for Repoducing

The original python VQE scripts runs on a deprecated version of `qiskit.algorithms` so it needs to be patched. The updated `.py` files are included in the repo. To setup the VQE algorithm integration:

```
python3 -m venv /path/to/virtual/environment
source /path/to/virtual/environment/bin/activate
pip install 'qiskit>=1'

# Install the other required packages
pip install qiskit-algorithms
pip install qiskit-aer

# Install the cp2k integration
cd qiskit-nature/qiskit_nature_cp2k
pip install .
```

There are also some known input file issues. In the VQE inputs for `MgO.inp`, in the `eri` section, the field for `poisson_solver` should be removed.

To reproduce the VQE results, the `client-vqe-ucc.py` and the `cp2k` program should be run in parallel on a single node so they can message pass via socket communication. 