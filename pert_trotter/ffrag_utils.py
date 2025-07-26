import numpy as np
import scipy as sp
from scipy.optimize import minimize
from opt_einsum import contract
from .tensor_utils import *
from copy import copy





def FRO_frags_generator(tbt, N_frags, ret_ops = True, ret_params = False):
  '''
  Generates Full rank fragments of a two body tensor using FRO method.

  Args:
      tbt (ndarray): two body tensor
      N_frags (int): Number of FR fragments to approximate tbt with.
      ret_ops(bool, optional): Return the fragments as FermionOperator objects. Defaults to True.
      ret_params(bool, optional): Return the parameters (lambda_ij and theta_ij) defining the fragments along with the fragments themself. Defaults to False.

  Returns:
      one or two lists: if ret_params == False -> if ret_ops == True -> list of FermmionOperator objects
                                                  else -> list of ndarrays corresponding to FR tbts
                        if ret_params == True ->  if ret_ops == True -> list of FermmionOperator objects, list of parameters
                                                  else -> list of ndarrays corresponding to FR tbts, list of parameters
  '''
  Htbt = copy(tbt)

  Htbt_spac = Htbt[np.ix_(*[range(0,i,2) for i in Htbt.shape])]
  N = Htbt_spac.shape[0]
  def cost(x):
    diff = copy(Htbt_spac)
    for n in range(N_frags):
      x_n = x[int(n*N**2):int((n+1)*N**2)]
      coeffs = x_n[:int(N*(N+1)/2)]
      angles = x_n[int(N*(N+1)/2):]
      FR_frag = build_FR_frag_tbt(coeffs, angles, N)
      diff -= FR_frag
    cost_val = np.sum(np.abs(diff*diff))
    print (cost_val)
    return cost_val

  x0 = np.random.uniform(-1e-3, 1e-3, size = N_frags*N**2)

  options = {
      'maxiter' : 10000,
      'disp'    : False
  }

  # tolerance
  tol     = 1e-4
  enum    = N ** 4
  fun_tol = (tol / enum) ** 2

  result = minimize(cost, x0, method = 'cobyla', tol=1e-7, options = options)
  sol = result.x

  params = []
  fragments = []
  for n in range(N_frags):
      x_n = sol[int(n*N**2):int((n+1)*N**2)]
      coeffs = x_n[:int(N*(N+1)/2)]
      angles = x_n[int(N*(N+1)/2):]
      params.append((coeffs, angles))

      FR_frag_spac = build_FR_frag_tbt(coeffs, angles, N)
      fragments.append(spac2spin(FR_frag_spac))

  if ret_ops == True:
    frag_ops = []
    for frag_tbt in fragments:
      frag_ops.append(tbt2op(frag_tbt))
    if ret_params == True:
      return frag_ops, params
    else:
      return frag_ops
  else:
    if ret_params == True:
      return fragments, params
    else:
      return fragments










def gfro_frag_params_optimized(Htbt, N, tol=1e-6):
  '''
  A helper function to gfro_frags_generator. It finds one FR fragment epsilon close in L1 norm distance to Htbt, where epsilon depends on tol.

  Args:
      Htbt (ndarray): two body tensor
      N (int): Htbt.shape[0]
      tol (float): Decides when to terminate optimization.

  Returns:
      ndarray: parameters defining the fragment; lambda_ij and theta_ij.
  '''
  def cost(x):
    coeffs = x[:int(N*(N+1)/2)]
    angles = x[int(N*(N+1)/2):]
    FR_frag = build_FR_frag_tbt(coeffs, angles, N)
    diff = Htbt - FR_frag
    cost_val = np.sum(np.abs(diff*diff))
    return cost_val

  x0 = np.random.uniform(-1e-3, 1e-3, size=N**2)

  options = {
      'maxiter' : 10000,
      'disp'    : False
  }

  # tolerance
  # tol     = 1e-5
  enum    = N ** 4
  fun_tol = (tol / enum) ** 2

  result = minimize(cost, x0, method = 'L-BFGS-B', tol=fun_tol, options = options)
  sol = result.x

  return sol








def gfro_frags_generator(tbt, numtag = 10000, tol = 1e-6, ret_ops = True, ret_params = False, spacial = False):
  """
  Generate GFRO fragments of a two body tensor.

  Args:
      Htbt (ndarray): two body tensor
      numtag (int, optional): Number of iterations of GFRO. Default is 10000.
      tol (float, optional): Decides when to terminate optimization in finding each fragment. Default is 1e-6.
      ret_ops (bool, optional): Return the fragments as FermionOperator objects. Defaults to True.
      ret_params (bool, optional): Return the parameters (lambda_ij and theta_ij) defining the fragments along with the fragments themself. Defaults to False
      spacial(bool, False): Whether to convert the tbt to spacial orbitals and find GFRO fragments. It provides ~16x improvement in speed. Defaults to False.

  Returns:
      one or two lists: if ret_params == False -> if ret_ops == True -> list of FermmionOperator objects
                                                  else -> list of ndarrays corresponding to FR tbts
                        if ret_params == True -> if ret_ops == True -> list of FermmionOperator objects, list of parameters
                                                  else -> list of ndarrays corresponding to FR tbts, list of parameters
  """
  Htbt = copy(tbt)
  N = Htbt.shape[0]

  if spacial == True:
    Htbt = Htbt[np.ix_(*[range(0,i,2) for i in Htbt.shape])]
    N = Htbt.shape[0]

  current_norm = np.sum(np.abs(Htbt * Htbt))
  fragments = []
  params = []

  for k in range(numtag):
    # end decomposition if remaining norm is less than 1e-6
    if current_norm < tol:
        break

    # obtain fragment parameters and remove fragment from Htbt
    sol   = gfro_frag_params_optimized(10*Htbt/current_norm, N, tol)
    coeffs = 0.1*current_norm*sol[:int(N*(N+1)/2)]
    angles = sol[int(N*(N+1)/2):]
    params.append((coeffs, angles))

    FR_frag = build_FR_frag_tbt(coeffs, angles, N)

    Htbt -= FR_frag
    if spacial == True:
      FR_frag = spac2spin(FR_frag)
    fragments.append(FR_frag)

    current_norm = np.sqrt(np.sum(np.abs(Htbt * Htbt)))

    print ('Current norm = ', current_norm)

  if ret_ops == True:
    frag_ops = []
    for frag_tbt in fragments:
      frag_ops.append(tbt2op(frag_tbt))
    if ret_params == True:
      return frag_ops, params
    else:
      return frag_ops
  else:
    if ret_params == True:
      return fragments, params
    else:
      return fragments













def sdgfro_frags_generator(obt, tbt, numtag = 10000, tol = 1e-4, ret_ops = True, ret_params = False, spacial = False):
  '''
  Find Singles-Doubles GFRO fragments of input {obt, tbt}.

  Args:
      obt (ndarray): one body tensor
      tbt (ndarray): two body tensor
      numtag (int, optional): Number of iterations of GFRO = 10000.
      tol (float, optional): Decides when to terminate optimization in finding each fragment.
      ret_ops (bool, optional): Return the fragments as FermionOperator objects. Defaults to True.
      ret_params (bool, optional): Return the parameters (two sets of lambda_ij and theta_ij) defining the fragments along with the fragments themself. Defaults to False
      spacial (bool, optional): Whether to convert the obt and tbt to spacial orbitals and find SDGFRO fragments. It provides ~16x improvement in speed. Defaults to False.

  Returns:
      one or two lists: if ret_params == False -> if ret_ops == True -> list of FermmionOperator objects
                                                  else -> list of ndarrays corresponding to FR tbts
                        if ret_params == True -> if ret_ops == True -> list of FermmionOperator objects, list of parameters_1, list of parameters_2
                                                  else -> list of ndarrays corresponding to FR tbts, list of parameters_1, list of parameters_2
  '''
  Htbt = copy(tbt)
  Htbt += obt2tbt(obt)

  idx_uuuu = np.ix_(*[range(0,i,2) for i in Htbt.shape])
  idx_dddd = np.ix_(*[range(1,i,2) for i in Htbt.shape])
  idx_uudd = np.ix_(*[range(0,n_qubits,2), range(0,n_qubits,2), range(1,n_qubits,2), range(1,n_qubits,2)])
  idx_dduu = np.ix_(*[range(1,n_qubits,2), range(1,n_qubits,2), range(0,n_qubits,2), range(0,n_qubits,2)])

  spac_tbt_1 = Htbt[idx_uuuu]
  spac_tbt_2 = Htbt[idx_uudd]
  N = spac_tbt_1.shape[0]
  spac_tbt_1norm, spac_tbt_2norm = np.sum(np.abs(spac_tbt_1 * spac_tbt_1)), np.sum(np.abs(spac_tbt_2 * spac_tbt_2))


  current_norm = 1
  fragments = []
  params_1 = []
  params_2 = []

  for k in range(numtag):
    # end decomposition if remaining norm is less than 1e-6
    if current_norm < tol:
        break

    # obtain fragment parameters and remove fragment from Htbt
    sol_1   = gfro_frag_params_optimized(10*spac_tbt_1/spac_tbt_1norm, N, tol)
    coeffs_1 = 0.1*spac_tbt_1norm*sol_1[:int(N*(N+1)/2)]
    angles_1 = sol_1[int(N*(N+1)/2):]
    params_1.append((coeffs_1, angles_1))

    sol_2   = gfro_frag_params_optimized(10*spac_tbt_2/spac_tbt_2norm, N, tol)
    coeffs_2 = 0.1*spac_tbt_2norm*sol_2[:int(N*(N+1)/2)]
    angles_2 = sol_2[int(N*(N+1)/2):]
    params_2.append((coeffs_2, angles_2))

    FR_frag_1 = build_FR_frag_tbt(coeffs_1, angles_1, N)
    FR_frag_2 = build_FR_frag_tbt(coeffs_2, angles_2, N)

    spac_tbt_1 -= FR_frag_1
    spac_tbt_2 -= FR_frag_2

    FR_frag_1_spin = np.zeros((2*N, 2*N, 2*N, 2*N), dtype = 'complex128')
    FR_frag_2_spin = np.zeros((2*N, 2*N, 2*N, 2*N), dtype = 'complex128')
    FR_frag_1_spin[idx_uuuu] = FR_frag_1_spin[idx_dddd] = FR_frag_1
    FR_frag_2_spin[idx_uudd] = FR_frag_2_spin[idx_dduu] = FR_frag_2
    fragments.append(FR_frag_1_spin)
    fragments.append(FR_frag_2_spin)

    spac_tbt_1norm, spac_tbt_2norm = np.sqrt(np.sum(np.abs(spac_tbt_1 * spac_tbt_1))), np.sqrt(np.sum(np.abs(spac_tbt_2 * spac_tbt_2)))
    current_norm = spac_tbt_1norm + spac_tbt_2norm
    print ('Current norm = ', current_norm)

  if ret_ops == True:
    frag_ops = []
    for frag_tbt in fragments:
      frag_ops.append(tbt2op(frag_tbt))
    if ret_params == True:
      return frag_ops, params_1, params_2
    else:
      return frag_ops
  else:
    if ret_params == True:
      return fragments, params_1, params_2
    else:
      return fragments














def gfro_frag_diag_params_optimized(Htbt, N):
  '''
  A Helper function to norm_supp_gfro_frags_generator.
  '''
  def cost(x):
    temp = x[:N]
    coeffs = np.zeros(int(N*(N+1)/2))
    diag_indxs = [0]+[int(i*(N-(i-1)/2)) for i in range (1, N)]
    for i in range (N):
      coeffs[diag_indxs[i]] = temp[i]

    angles = x[N:]
    FR_frag = build_FR_frag_tbt(coeffs, angles, N)
    diff = Htbt - FR_frag
    cost_val = np.sum(np.abs(diff*diff))
    return cost_val

  x0 = np.random.uniform(-1e-3, 1e-3, size=int(N*(N+1)/2))

  options = {
      'maxiter' : 10000,
      'disp'    : False
  }

  # tolerance
  tol     = 1e-5
  enum    = N ** 4
  fun_tol = (tol / enum) ** 2

  result = minimize(cost, x0, method = 'cobyla', tol=fun_tol, options = options)
  sol = result.x

  return sol







def gfro_frag_nondiag_params_optimized(Htbt, N, diag_coeffs, angles):
  '''
  A Helper function to norm_supp_gfro_frags_generator.
  '''
  def cost(x, diag_coeffs, angles):
    coeffs = np.zeros(int(N*(N+1)/2))
    diag_indxs = [0]+[int(i*(N-(i-1)/2)) for i in range (1, N)]
    diag_indxs += [diag_indxs[-1]]
    k = 0
    for i in range (N):
      coeffs[diag_indxs[i]] = diag_coeffs[i]
      for j in range (diag_indxs[i]+1, diag_indxs[i+1]):
        coeffs[j] = x[k]
        k += 1

    FR_frag = build_FR_frag_tbt(coeffs, angles, N)
    diff = Htbt - FR_frag
    cost_val = np.sum(np.abs(diff*diff))
    return cost_val

  x0 = np.random.uniform(-1e-3, 1e-3, size=int(N*(N-1)/2))

  options = {
      'maxiter' : 10000,
      'disp'    : False
  }

  # tolerance
  tol     = 1e-5
  enum    = N ** 4
  fun_tol = (tol / enum) ** 2

  result = minimize(cost, x0, args=(diag_coeffs, angles), method = 'cobyla', tol=fun_tol, options = options)
  sol = result.x

  return sol









def norm_supp_gfro_frags_generator(tbt, numtag = 100, tol = 1e-4, ret_ops = True, ret_params = False):
  """
  A new way to find GFRO fragments, where the first fragment is found by optimizing the diagonal and non diagonal elements of tbt separately. Rest of the fragments are found as usual GFRO.
  This was to check if we can concentrate all the one electron terms of GFRO fragments into a single fragment.

  Args:
      Htbt (ndarray): two body tensor
      numtag (int, optional): Number of iterations of GFRO = 10000.
      tol (float, optional): Decides when to terminate optimization in finding each fragment. Default is 1e-6.
      ret_ops (bool, optional): Return the fragments as FermionOperator objects. Defaults to True.
      ret_params (bool, optional): Return the parameters (lambda_ij and theta_ij) defining the fragments along with the fragments themself. Defaults to False

  Returns:
      one or two lists: if ret_params == False -> if ret_ops == True -> list of FermmionOperator objects
                                                  else -> list of ndarrays corresponding to FR tbts
                        if ret_params == True -> if ret_ops == True -> list of FermmionOperator objects, list of parameters
                                                  else -> list of ndarrays corresponding to FR tbts, list of parameters
  """
  Htbt = copy(tbt)
  N = Htbt.shape[0]
  current_norm = 1
  fragments = []
  params = []

  for l in range(numtag):
    # end decomposition if remaining norm is less than 1e-6
    if current_norm < tol:
        break

    if l == 0:
      print ('Method 1')
      # obtain fragment parameters and remove fragment from Htbt
      sol_1  = gfro_frag_diag_params_optimized(Htbt, N)
      diag_coeffs = sol_1[:N]
      angles = sol_1[N:]

      sol_2  = gfro_frag_nondiag_params_optimized(Htbt, N, diag_coeffs, angles)
      coeffs = np.zeros(int(N*(N+1)/2))
      diag_indxs = [0]+[int(i*(N-(i-1)/2)) for i in range (1, N)]
      diag_indxs += [diag_indxs[-1]]
      k = 0
      for i in range (N):
        coeffs[diag_indxs[i]] = diag_coeffs[i]
        for j in range (diag_indxs[i]+1, diag_indxs[i+1]):
          coeffs[j] = sol_2[k]
          k += 1
    else:
      print ('Method 2')
      # obtain fragment parameters and remove fragment from Htbt
      sol   = gfro_frag_params_optimized(Htbt, N)
      coeffs = sol[:int(N*(N+1)/2)]
      angles = sol[int(N*(N+1)/2):]

    params.append((coeffs, angles))

    FR_frag = build_FR_frag_tbt(coeffs, angles, N)

    Htbt -= FR_frag
    fragments.append(FR_frag)

    current_norm = np.sum(np.abs(Htbt * Htbt))

    print ('Current norm = ', current_norm)

  if ret_ops == True:
    frag_ops = []
    for frag_tbt in fragments:
      frag_ops.append(tbt2op(frag_tbt))
    if ret_params == True:
      return frag_ops, params
    else:
      return frag_ops
  else:
    if ret_params == True:
      return fragments, params
    else:
      return fragments










def LR_frags_generator(Htbt, tol = 1e-6, ret_params = True, spacial = True):
  """
  Generate low-rank fragments of a two body tensor (by exactly daigonalizing tbt supermatrix).

  Args:
      Htbt (ndarray): two body tensor
      tol (float, optional): LR fragments with coefficients less tol will be discarded. Default values to 1e-6.
      ret_params (bool, optional): Return the parameters (lambda_ij and theta_ij) defining the fragments along with the fragments themself. Defaults to False
      spacial(bool, False): Whether to convert the tbt to spacial orbitals and find GFRO fragments. It provides ~16x improvement in speed. Defaults to False.

  Returns:
      one or two lists: if ret_params == False -> list of FermmionOperator objects
                        if ret_params == True -> list of FermmionOperator objects, list of parameters
  """
  if spacial == True:
    Htbt = Htbt[np.ix_(*[range(0,i,2) for i in Htbt.shape])]
  N = Htbt.shape[0]
  sup_mat = Htbt.reshape((N**2, N**2))
  cur_Ds, cur_Ls = np.linalg.eigh(sup_mat)
  Ls = [cur_Ls[:,i].reshape((N,N)) for i in range (len(cur_Ls))]

  LR_fragments = []
  params = []

  for i in range (len(Ls)):
    L = Ls[i]
    cur_D = cur_Ds[i]

    if np.linalg.norm(np.sqrt(np.abs(cur_D))*L)>tol:
      # frag  = FermionOperator()
      # for p in range(N):
      #   for q in range(N):
      #     term = ((p,1), (q,0))
      #     coef = L[p,q]
      #     frag   += FermionOperator(term, coef)

      # LR_fragments.append(cur_D*frag*frag)

      d, u = np.linalg.eigh(L)
      d = d.reshape((len(d), 1))
      coeff_mat = cur_D*d@d.T
      params.append((coeff_mat, u))

      frag_tbt = build_FR_frag_tbt_ez(coeff_mat, u)
      if spacial == True:
        frag_tbt = spac2spin(frag_tbt)
      LR_fragments.append(tbt2op(frag_tbt))


  if ret_params == True:
    return LR_fragments, params
  else:
    return LR_fragments












def LCU_largest_frag(Htbt, tol = 1e-6):                           #Get a mean field solvable fragment with largest one-norm
  N = Htbt.shape[0]
  c, u, p = num_params(N)

  def cost(x):
    frag_tbt = get_fragment(x, N)
    diff = Htbt - frag_tbt
    _, diff_op, _ = chem_ten2op(np.zeros((N, N)), diff, N)
    diff_JW = jordan_wigner(normal_ordered(diff_op))

    norm = 0
    for key, val in diff_JW.terms.items():
      if np.abs(val)>tol:
        norm+=np.abs(val)
    print (norm)
    return norm

  # initial guess
  x0       = np.ones(p)
  # x0 = np.zeros(p)
  x0[c : ] = np.random.uniform(-np.pi/2, np.pi/2, u)
  # x0[c : ] = np.zeros(u)

  # options
  options = {
      'maxiter' : 10000,
      'disp'    : False
  }

  # tolerance
  enum    = N ** 4
  fun_tol = (tol / enum) ** 2

  # optimize
  result = minimize(cost, x0, method='COBYLA', tol=fun_tol, options=options)
  final_frag = get_fragment(result.x, N)
  return final_frag









def get_coeff_mat_from_coeffs(coeffs, N):
  '''
  Convert the 1D list of parameters defining a FR fragment to a matrix lambda_ij

  Args:
      coeffs (list or 1d np.array): Coefficients as a vector
      N (int): Number of spin/spacial orbitals in the fragment

  Returns:
      np.array: Coefficient matrix lambda_ij. Shape = (N,N).
  '''
  coeff_mat = np.zeros((N,N), dtype = 'complex128')
  idx = 0
  for i in range (N):
    for j in range (i,N):
      coeff_mat[i,j] = coeffs[idx]
      coeff_mat[j,i] = coeffs[idx]
      idx += 1
  return coeff_mat








def get_u_from_angles(angles, N):
  '''
  Convert the 1D list of parameters defining orbital rotation to matrix u. We want u=exp(K), where
  K is an anti-hermitian matrix defined by the parameters 'angles'.

  Args:
      angles (list or 1d np.array): Angles as a vector
      N (int): Number of spin/spacial orbitals in the fragment

  Returns:
      np.array: Orbital rotation matrix u. Shape = (N,N).
  '''
  kappa = np.zeros((N, N), dtype = 'complex128')
  idx=0
  for p in range (N-1):
    for q in range (p+1, N):
      kappa[p,q] = angles[idx]
      kappa[q,p] = -angles[idx]
      idx += 1
  u = sp.linalg.expm(kappa)
  return u









def build_FR_frag_tbt(coeffs, angles, N = None):
  '''
  Build FR tbt from coefficients and angles defining the tbt.
  len(coeffs) = N(N+1)/2. len(angles) = N(N-1)/2.

  Args:
      coeffs (list or 1d np.array): Coefficients as a vector
      angles (list or 1d np.array): Angles as a vector
      N (int, optional): Number of spin/spacial orbitals in the fragment. Defaults to None, in which case it equals global parameter n_qubits.

  Returns:
      np.array: chemist two-body-tensor. Shape = (N,N,N,N).
  '''
  if N == None:
    N = n_qubits
  coeff_mat = get_coeff_mat_from_coeffs(coeffs, N)
  u = get_u_from_angles(angles, N)
  return contract('ij,pi,qi,rj,sj -> pqrs', coeff_mat, u, u, u, u)









def build_FR_frag_tbt_frmo_so(coeffs, angles, N = None): #len(coeffs) = N(N+1)/2. len(angles) = N(N-1)/2. N = number of spin orbitals = n_qubits
  '''
  Essentially same as build_FR_frag_tbt. So use that instead of this.
  '''
  if N == None:
    N = n_qubits

  coeff_mat = np.zeros((N,N), dtype = 'complex128')
  idx = 0
  for i in range (N):
    for j in range (i,N):
      coeff_mat[i,j] = coeffs[idx]
      coeff_mat[j,i] = coeffs[idx]
      idx += 1

  kappa = np.zeros((N, N), dtype = 'complex128')
  idx=0
  for p in range (N-1):
    for q in range (p+1, N):
      kappa[p,q] = angles[idx]
      kappa[q,p] = -angles[idx]
      idx += 1
  u = sp.linalg.expm(kappa)

  return contract('ij,pi,qi,rj,sj -> pqrs', coeff_mat, u, u, u, u)





def gfro_spac_param2FOP(coeffs, angles, N_spac):
  '''
  Convert coeffs, angles that define a FR fragment in spacial orbitals to a FermionOperator in spin orbitals.

  Args:
      coeffs (list or 1d np.array): Coefficients as a vector
      angles (list or 1d np.array): Angles as a vector
      N_spac (int): Number of spacial orbitals in the fragment

  Returns:
      FermionOperator: FermionOperator in spin orbitals corresponding to the input FR fragment.
  '''
  spac_tbt = build_FR_frag_tbt(coeffs, angles, N_spac)
  spin_tbt = spac2spin(spac_tbt)
  return tbt2op(spin_tbt)







def build_FR_frag_tbt_ez(coeff_mat, u):
  '''
  Convert coefficient matrix and orbital rotation unitary defining a FR fragment to tbt in chemist ordering.

  Args:
      coeff_mat (np.array): Coefficient matrix lambda_ij. Shape = (N,N).
      u (np.array): Orbital rotation matrix u. Shape = (N,N).

  Returns:
      np.array: chemist two-body-tensor. Shape = (N,N,N,N).
  '''
  return contract('ij,pi,qi,rj,sj -> pqrs', coeff_mat, u, u, u, u)

