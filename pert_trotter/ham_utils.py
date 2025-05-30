import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigsh
import math
from openfermion import FermionOperator, QubitOperator, MolecularData, hermitian_conjugated, get_molecular_data
from openfermion.transforms import get_fermion_operator
from openfermionpyscf import run_pyscf






def obtain_OF_hamiltonian(mol_name, ferm = True, basis='sto3g', geometry=1, n_elec = True):
  '''
  Generate molecular Hamiltonian in OpenFermion format. If ferm == True --> Return the opeartor as FermionOperator object.
  '''
  g = chooseType(mol_name, geometry)
  mol = MolecularData(g, basis, 1, 0)
  mol = run_pyscf(mol)
  ham = mol.get_molecular_hamiltonian()
  if n_elec == False:
      if ferm:
          return get_fermion_operator(ham)
      else:
          return ham
  else:
      if ferm:
          return get_fermion_operator(ham), mol.n_electrons
      else:
          return ham, mol.n_electrons








def chooseType(typeHam, geometries):
    '''
    Genreate the molecular data of specified type of Hamiltonian
    '''
    if typeHam == 'h2':
        molData = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometries]]
        ]
    elif typeHam == 'h3':
        molData = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometries]],
            ['H', [0, 0, 2*geometries]]
        ]
    elif typeHam == 'h4_chain':
        molData = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometries]],
            ['H', [0, 0, 2*geometries]],
            ['H', [0, 0, 3*geometries]]
        ]
    elif typeHam == 'h6_chain':
        molData = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometries]],
            ['H', [0, 0, 2*geometries]],
            ['H', [0, 0, 3*geometries]],
            ['H', [0, 0, 4*geometries]],
            ['H', [0, 0, 5*geometries]]
        ]
    elif typeHam == 'n2':
        molData = [
            ['N', [0, 0, 0]],
            ['N', [0, 0, geometries]]
        ]
    elif typeHam == 'lih':
        molData = [
            ['Li', [0, 0, 0]],
            ['H', [0, 0, geometries]]
        ]
    # Giving symmetrically stretch H2O. ∠HOH = 104.5°
    elif typeHam == 'h2o':
        angle = 104.5 / 2
        angle = math.radians(angle)
        xDistance = geometries * math.sin(angle)
        yDistance = geometries * math.cos(angle)
        molData = [
            ['O', [0, 0, 0]],
            ['H', [-xDistance, yDistance, 0]],
            ['H', [xDistance, yDistance, 0]]
        ]
    elif typeHam == 'n2':
        molData = [
            ['N', [0, 0, 0]],
            ['N', [0, 0, geometries]]
        ]
    elif typeHam == 'beh2':
        molData = [
            ['Be', [0, 0, 0]],
            ['H', [0, 0, -geometries]],
            ['H', [0, 0, geometries]]
        ]
    elif typeHam == 'h4':
        angle1 = math.radians(geometries/2)
        angle2 = math.radians(90-geometries/2)
        R = 1.737236
        hor_val = 2*R*math.sin(angle1)
        ver_val = 2*R*math.sin(angle2)
        molData = [
            ['H', [0, 0, 0]],
            ['H', [hor_val, 0, 0]],
            ['H', [0, ver_val, 0]],
            ['H', [hor_val, ver_val, 0]]
        ]
    elif typeHam == 'nh3':
    # Is there a more direct way of making three vectors with specific mutual angle?
        bondAngle = 107
        bondAngle = math.radians(bondAngle)
        cos = math.cos(bondAngle)
        sin = math.sin(bondAngle)

        # The idea is second and third vector dot product is cos(angle) * geometry^2.
        thirdyRatio = (cos - cos**2) / sin
        thirdxRatio = (1 - cos**2 - thirdyRatio**2) ** (1/2)
        molData = [
            ['H', [0, 0, geometries]],
            ['H', [0, sin * geometries, cos * geometries]],
            ['H', [thirdxRatio * geometries, thirdyRatio * geometries, cos * geometries]],
            ['N', [0, 0, 0]],
        ]
    else:
        raise(ValueError(typeHam, 'Unknown type of hamiltonian given'))

    return molData











def spec_range(H_JW_Sparse, tol = 1e-5):
  '''
  Calculate the spectral range of the input Hamiltonian.

  Args:
      H_JW_Sparse (scipy.sparse.csc_matrix): Input Hamiltoian whose specrtal range is needed.
      tol (optional, float): Input to scipy.sparse.eigsh function

  Returns:
      float: Spectral range of the input Hamiltonian.
  '''
  E_min = eigsh(H_JW_Sparse, k = 1, which='SA', tol = tol, return_eigenvectors=False)
  E_max = eigsh(H_JW_Sparse, k = 1, which='LA', tol = tol, return_eigenvectors=False)
  return E_max[0]-E_min[0]