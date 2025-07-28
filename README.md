# Perturbative_Trotter_Error

The repository contains code to implement research done in the work [Estimating Trotter Approximation Errors to Optimize Hamiltonian Partitioning for Lower Eigenvalue Errors](https://arxiv.org/abs/2312.13282). We build functionalities to generate fragments of molecular Hamiltonians, calculate exact and approximate Trotter error based on perturbation theory, and estimate T gate counts for implementing Quantum Phase Estimation using the second order Trotter approximation. The source code is contained in the package ```pert_trotter```. The google colab notebook ```tutorial.ipynb``` provides a step-by-step guideline to reproduce the results in the paper. It can also be opened in Colab from [here](https://colab.research.google.com/github/Shashank-G-M/Perturbative_Trotter_Error/blob/main/tutorial.ipynb). We make extensive use of [OpenFermion](https://github.com/quantumlib/OpenFermion) and [PySCF](https://github.com/pyscf/pyscf) to constuct molecular Hamiltonians. Note, for the qubitized fragments of molecules BeH2, H2O, and NH3, a compute cluster might be needed to obtain some of the quantities mentioned in the paper.

Code to an older version of the paper developed by [@prathami11](https://github.com/prathami11) that analyzes first order Trotter approximation error can be found [here](https://github.com/prathami11/TrueTrotterError).

## Author

[@Shashank-G-M](https://github.com/Shashank-G-M)
