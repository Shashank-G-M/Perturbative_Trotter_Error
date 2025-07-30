import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigsh
from itertools import combinations
from openfermion import QubitOperator, get_sparse_operator, get_number_preserving_sparse_operator
from openfermion import jw_hartree_fock_state, jordan_wigner
from openfermion.hamiltonians import s_squared_operator, sz_operator, number_operator
from copy import copy





def occ_str_to_state(binary, ret_sparse  =False):
  '''
  Convert slater determinant state from a string (eg: '11100') or a binary (eg: 11100) to an array in Fock space.
  For example: 11100 gets mapped to a 2^5 x 1 dimensional zero array with one at row index = 2^0 + 2^1 + 2^2 - 1.
  Left most state = inner most spin orbital.

  Args:
      binary (str or int): Slater determinant represented as a string or a binary.
      ret_sparse (bool, optional): If True, the returned array would be of type scipy.sparse.csc_matrix. Default is False.

  Returns:
      array: if ret_sparse == True -->  scipy.sparse.csc_matrix of dimension 2^len(binary) x 1.
                              else --> numpy.ndarray of dimension 2^len(binary) x 1.
  '''
  binary = str(binary)
  decimal = int(binary, 2)
  if ret_sparse == False:
    occ_vec = np.zeros((2**len(binary), 1))
    occ_vec[decimal, 0] = 1
  else:
    occ_vec = sp.sparse.csc_matrix.zeros((2**len(binary), 1))
    occ_vec[decimal, 0] = 1
  return occ_vec









def occ_str_to_pauli_x(binary):
  """
  Given a slater determinant written as a string or a binary, return a qubit
  operator that gives rise to the slater determinant when acted on vacuum state.
  For example: passing 1101 or '1101' gives QubitOperator('X0 X1 X3').

  Args:
      binary (str or int): Slater determinant represented as a string or a binary.

  Returns:
      QubitOperator: QubitOperator that generates the slater determinant corresponding to binary when acted on vacuum.
  """
  binary = str(binary)
  pauli_string = ' '.join('X'+str(i) for i in range(len(binary)) if binary[i] == '1')
  return QubitOperator(pauli_string, 1)








def act_Q_OP_on_vacuum(Q_OP, n_qubits, eps = 1e-8):
  """
  Given a QubitOperator and the number of qubits, return the action of the QubitOperator
  on the vacuum state. Output is a dictionary with binary string of slater determinants
  as keys and the corresponding coefficients as values.

  Args:
      Q_OP (QubitOperator): Input QubitOperator that acts on vacuum.
      n_qubits (int): Number of qubits.

  Returns:
      dict: Dictionary with slater determinants as binary strings as keys and corresponding coefficients as values.
  """
  output = {}
  for word, coeff in Q_OP.terms.items():
    new_state = list('0'*n_qubits)
    val = coeff
    for letter in word:
      if letter[1] == 'X':
        new_state[letter[0]] = '1'
      elif letter[1] == 'Y':
        new_state[letter[0]] = '1'
        val *= 1.j
    new_state = ''.join(new_state)
    if new_state in output:
      output[new_state] += val
    else:
      output[new_state] = val
  output = {key:val for key, val in output.items() if np.abs(val) > eps}
  return output











def get_nth_excitations(ref_det, order = int):
  '''
  Generate all possible exitations upto rank = order.

  Args:
      ref_det (str or int): reference determinant represented by a string or binary. Eg: '1100' or 1100.
      order (int): Rank up to which exitations are generated. 0 = No exitation. 1 = Singles. 2 = Doubles.

  Returns:
      list: List of all exitations upto rank = order represented as strings. Includes ref_det.
  '''
  ref_det = str(ref_det)
  n = order
  all_vecs = []
  if n == 0:
    return [ref_det]

  occ_indx = [i for i in range(0, len(ref_det)) if ref_det[i] == '1']
  empty_indx = [i for i in range(0, len(ref_det)) if ref_det[i] == '0']

  for indx_1 in combinations(occ_indx, n):
    for indx_2 in combinations(empty_indx, n):
      ex_det = np.array(list(copy(ref_det)))
      ex_det[list(indx_1)], ex_det[list(indx_2)] = ex_det[list(indx_2)], ex_det[list(indx_1)]
      all_vecs.append(''.join(ex_det))
  return all_vecs






def get_Sz_preserving_nth_excitations(ref_det, order = int):
  '''
  Generate all exitations upto rank = order that preserve the Sz value of the ref_det.

  Args:
      ref_det (str or int): reference determinant represented by a string or binary. Eg: '1100' or 1100.
      order (int): Rank up to which exitations are generated. 0 = No exitation. 1 = Singles. 2 = Doubles.

  Returns:
      list: List of all exitations upto rank = order that preserve Sz value of ref_det, represented as strings. Includes ref_det.
  '''
  ref_det = str(ref_det)
  n = order
  all_vecs = []
  if n == 0:
    return [ref_det]
  occ_indx_alpha = [i for i in range(0, len(ref_det), 2) if ref_det[i] == '1']
  occ_indx_beta = [i for i in range(1, len(ref_det), 2) if ref_det[i] == '1']
  empty_indx_alpha = [i for i in range(0, len(ref_det), 2) if ref_det[i] == '0']
  empty_indx_beta = [i for i in range(1, len(ref_det), 2) if ref_det[i] == '0']

  for i in range (min(n, len(occ_indx_alpha))+1):
    a = i
    b = min(n-a, len(occ_indx_beta))
    if (a+b < n) or (a > len(empty_indx_alpha)) or (b > len(empty_indx_beta)):
      pass
    else:
      for indx_0 in combinations(occ_indx_alpha, a):
        for indx_1 in combinations(occ_indx_beta, b):
          for indx_2 in combinations(empty_indx_alpha, a):
            for indx_3 in combinations(empty_indx_beta, b):
              HF_array = np.array(list(ref_det))
              HF_array[list(indx_0)], HF_array[list(indx_2)] = HF_array[list(indx_2)], HF_array[list(indx_0)]
              HF_array[list(indx_1)], HF_array[list(indx_3)] = HF_array[list(indx_3)], HF_array[list(indx_1)]
              all_vecs.append(''.join(HF_array))
  if len(all_vecs) == 0:
    print ('No compatible excitations are possible for the given excitation order and the reference determinant')
  return list(set(all_vecs))









def get_CI_proj_ham(H_JW_sarray, n_excitations=int, ref_state = None, ret_excitations = False, num_elecs = int, n_qubits = int): #n stands for CI order. That is, n = 1 => CIS, n=2 => CISD etc.
  '''
  Obtain the projection of input Hamiltonian in the configuration interaction subspace of rank = n_excitations.
  The assumption here is that, the input array is expressed in the basis of slater determinants, and the order
  of the determinants is basically the increasing order of the decimal value of the binary representation of the
  determinant. For example, the state |00000> is the basis function at index 0, |00001> at index 1, |00011> at
  index 3, etc.

  Args:
      H_JW_sarray (scipy.sparse.csc_matrix): Hamiltonian as Scipy csc_matrix expressed in the basis of slater determiannts.
      n_excitations (int): Rank of the CI exitations. If equals 1 => CIS, if equals 2 => CISD, etc.
      ref_state (str or int): String or binary representation of reference state needed for excitations. If None, assumes HF state
                              obtained by global variables n_qubits and num_elecs.
      ret_excitations (bool, optional): If true, returns the list of excitations along with the projected Hamiltonian.
      num_elecs = Number of electrons in the system
      n_qubits = Number of qubits in the system

  Returns:
      scipy.sparse.csc_matrix: Projected Hamiltonian.
      (if ret_excitations == True) list:  --> list of excitations forming the basis of the projected Hamiltonian.
  '''
  if ref_state == None:
    n_occ = num_elecs
    n_empty = n_qubits - n_occ
  else:
    n_occ = sum(int(i) for i in ref_state)
    n_empty = len(ref_state) - n_occ

  occ_indx = range(n_occ)
  empty_indx = range(n_occ, n_qubits)

  ref_state = '1'*n_occ+'0'*n_empty       #ref_state = Hartree-Fock state

  all_excitations = [ref_state]

  for n in range (1,n_excitations+1):
    all_excitations += get_nth_excitations(ref_state, n)

  all_excitations_in_decimal = []
  for i in range(len(all_excitations)):
    binary = str(all_excitations[i])
    decimal = int(binary, 2)
    all_excitations_in_decimal.append(decimal)

  proj_H = np.zeros((len(all_excitations), len(all_excitations)))
  for i in range(len(all_excitations)):
    dec_1 = all_excitations_in_decimal[i]
    for j in range(i, len(all_excitations)):
      dec_2 = all_excitations_in_decimal[j]
      proj_H[i,j] = proj_H[j,i] = np.real(H_JW_sarray[dec_1,dec_2])
  if ret_excitations:
    return proj_H, all_excitations
  else:
    return proj_H









def verify_HF_idx_in_NSzproj_space(H, Hchem, n_qubits, num_elecs):
  '''
  Verifies if <HF|H|HF> = H_{proj}[0,0], where H_proj is the Hamiltonian projected on to the Sz = 0 and N = num_elecs subspace.
  This should conclude that, after using get_number_preserving_sparse_operator, index 0 in the set of basis states corresponds
  to the reference determinant, i.e., HF state.

  Args:
      H (FermionOpeartor): Hamiltonian.
      Hchem (FermionOpeartor): Hamiltonian in the projected space.
      n_qubits (int): Number of qubits.
      num_elecs (int): Number of electrons.

  Returns:
      None: It just prints the value of <HF|H|HF> - H_{proj}[0,0].
  '''
  if n_qubits > 14:
    print ('Number of qubits should be less than or equal to 14.')
    return None
  H_JW = jordan_wigner(H)
  H_JW_sarray = get_sparse_operator(H_JW, n_qubits)
  HF_array = jw_hartree_fock_state(num_elecs, n_qubits).reshape(2**n_qubits, 1)
  HF_sarray = sp.sparse.csc_matrix(HF_array)
  val_1 = (HF_sarray.T*H_JW_sarray*HF_sarray)[0,0]

  H_proj_sarray = get_number_preserving_sparse_operator(Hchem, n_qubits, num_elecs, spin_preserving=True)
  val_2 = H_proj_sarray[0,0]
  print ('The value of |<HF|H|HF> - H_{proj}[0,0]| = ', np.abs(val_1-val_2))
  return None









def eval_GS_HF_overlap(H, n_qubits, num_elecs, verify_Ssq_is_0 = True, tol = 1e-6):
  '''
  Evaluates the overlap of ground state of H in the correct Sz, N subspace with the HF state. The asumption is that the
  basis in which H is expressed is the usual slater determinant basis.

  Args:
      H (FermionOpeartor): Hamiltonian.
      n_qubits (int): Number of qubits.
      num_elecs (int): Number of electrons.
      verify_Ssq_is_0 (bool, optional): If true, prints if |<GS|S^2|GS>| < tol or not. Defaults to True.
      tol (optional, float): Tolerance to determine if expval of S^2 is zero or not. Defaults to 1e-6.

  Returns:
      float: Overlap of ground state with HF state.
  '''
  H_proj_sarray = get_number_preserving_sparse_operator(H, n_qubits, num_elecs, spin_preserving=True)
  v0, w0 = eigsh(H_proj_sarray, k = 1, which='SA', return_eigenvectors=True)
  w0_sarray = sp.sparse.csc_matrix(w0)
  if verify_Ssq_is_0 == True:
    Ssq = s_squared_operator(n_qubits//2)
    Ssq_sarray = get_number_preserving_sparse_operator(Ssq, n_qubits, num_elecs, spin_preserving=True)
    expval = np.abs(w0_sarray.T*Ssq_sarray*w0_sarray)[0,0]
    if np.abs(expval) > tol:
      print ('Expectation value wrt S^2 is not zero')
    else:
      print ('Expectation value wrt S^2 is zero')
  return w0_sarray[0,0]








def eval_GS_CISD_overlap(JW_OF_sarray, n_qubits, num_elecs, verify_Sz_N_Ssq_evals = True):
  '''
  Given JW_OF_sarray, return the abs(overlap)^2 of the ground state with the CISD approximation to the ground state.
  Also can verify if Sz = 0, N = num_elecs and S^2 = 0 for the CISD ground state.

  Args:
      JW_OF_sarray (Scipy.sparse.scs_matrix): Input Hamiltonian as sparse matrix
      n_qubits (int): Number of qubits
      num_elecs (int): Number of electrons
      verify_Sz_N_Ssq_evals (bool, optional): If True, checks if Sz = 0, N = num_elecs and S^2 = 0
                                              for the CISD ground state. Defaults to True.

  Returns:
      float: Overlap of the ground state with the CISD ground state.
  '''
  v0, w0 = eigsh(np.real(JW_OF_sarray), k = 1, which='SA', return_eigenvectors=True)
  w0_sparse = sp.sparse.csc_matrix(w0)

  proj_H, all_excitations = get_CI_proj_ham(JW_OF_sarray, n_excitations = 2, n_qubits = n_qubits, num_elecs = num_elecs, ret_excitations = True)
  all_excitations_vec = np.zeros((2**n_qubits, len(all_excitations)))
  for i in range(len(all_excitations)):
    all_excitations_vec[:,[i]] = occ_str_to_state(all_excitations[i])
  CISD_evals,CISD_evecs = np.linalg.eigh(proj_H)
  CISD_evecs_full_space = all_excitations_vec@CISD_evecs
  CISD_w0_sparse = sp.sparse.csc_matrix(CISD_evecs_full_space[:,[0]])

  if verify_Sz_N_Ssq_evals == True:
    Ssq = s_squared_operator(n_qubits//2)
    Ssq_full_sparse = get_sparse_operator(Ssq, n_qubits)
    Ssq_expval = np.abs(CISD_w0_sparse.T*Ssq_full_sparse*CISD_w0_sparse)[0,0]
    if np.abs(Ssq_expval) > 1e-6:
      print ('S^2 is not zero')
    else:
      print ('S^2 is zero')
    Sz = sz_operator(n_qubits//2)
    Sz_full_sparse = get_sparse_operator(Sz, n_qubits)
    Sz_expval = np.abs(CISD_w0_sparse.T*Sz_full_sparse*CISD_w0_sparse)[0,0]
    if np.abs(Sz_expval) > 1e-6:
      print ('Sz is not zero')
    else:
      print ('Sz is zero')
    N_op = number_operator(n_qubits)
    N_full_sparse = get_sparse_operator(N_op, n_qubits)
    N_expval = np.abs(CISD_w0_sparse.T*N_full_sparse*CISD_w0_sparse)[0,0]
    if np.abs(N_expval - num_elecs) > 1e-6:
      print ('N is not equal to num_elecs')
    else:
      print ('N is equal to num_elecs')

  overlap = (w0_sparse.T*CISD_w0_sparse)[0,0]
  return np.abs(overlap)**2







