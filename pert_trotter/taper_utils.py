import numpy as np
import scipy as sp
from openfermion import QubitOperator
from .fock_utils import *











def gen_tap_unitary(n_qubits):
  """
  Generate unitary transformation to taper qubits. The symmetries used are Parity operators
  of alpha and beta spin flavours. Returns the unitary U, and its Hermitian conjugate U_dag as QubitOperators.
  Then to store the parities in the last two qubits, use the unitaries on a qubit operator JW_OF as U*JW_OF*U_dag.
  If every term of JW_OF commutes with the parity operators, then each pauli word in the resultant QubitOperator
  will have Zs on the last two qubits. Then to finally taper JW_OF, replace the two Zs by +/-1 depending  on the
  subspace of interest. For example, if we know the ground state has odd number of alpha and beta electrons then
  replace both Zs by -1. If however both are even, replace them by +1.

  Args:
      n_qubits (int): Number of qubits.

  Returns:
      QubitOperator: U.
      QubitOperator: U_dag.

  """
  alpha_parity_Q_OP = QubitOperator(' '.join('Z%i'%i for i in range(0, n_qubits, 2)))
  alpha_parity_posn = n_qubits - 2

  U1 = (QubitOperator(f'Z{alpha_parity_posn}') + QubitOperator(f'X{alpha_parity_posn}'))/np.sqrt(2)
  U2 = (QubitOperator(f'X{alpha_parity_posn}') + alpha_parity_Q_OP)/np.sqrt(2)

  beta_parity_Q_OP = QubitOperator(' '.join('Z%i'%i for i in range(1, n_qubits, 2)))
  beta_parity_posn = n_qubits - 1
  U3 = (QubitOperator(f'Z{beta_parity_posn}') + QubitOperator(f'X{beta_parity_posn}'))/np.sqrt(2)
  U4 = (QubitOperator(f'X{beta_parity_posn}') + beta_parity_Q_OP)/np.sqrt(2)

  U = U2*U1*U4*U3
  U_dag = U3*U4*U1*U2

  return U, U_dag











def taper_qubits(JW_OF, n_qubits, num_alphas, num_betas):
  """
  Given a qubit operator with each of its terms commuting with parity operators, return a tapered qubit operator.
  The num_alphas and num_betas decide which subspace to select. For example, if num_alphas is odd and num_betas
  is even, then the subspace chosen will replace any Z's acting on the last but one and last qubits by -1 and +1
  respectively.

  Args:
      JW_OF (QubitOperator): QubitOperator that needs to be tapered.
      n_qubits (int): Number of qubits.
      num_alphas (int): Number of alpha electrons in the subspace of interest.
      num_betas (int): Number of beta electrons in the subspace of interest.

  Returns:
      QubitOperator: QubitOperator that is tapered. It has 2 less qubits than the input Hamiltonian.
  """
  U, U_dag = gen_tap_unitary(n_qubits)
  JW_OF_par_rotated = U*JW_OF*U_dag
  JW_OF_par_rotated.compress(1e-9)

  alpha_parity_posn = n_qubits - 2
  alpha_coeff = 1 if num_alphas%2 == 0 else -1

  beta_parity_posn = n_qubits - 1
  beta_coeff = 1 if num_betas%2 == 0 else -1

  JW_OF_tapered = QubitOperator.zero()
  for term, coeff in JW_OF_par_rotated.terms.items():
    if len(term) != 0:
      if term[-1][0] == beta_parity_posn:
        if (term[-1][1] == 'Z'):
          term = term[:-1]
          coeff *= beta_coeff
        else:
          raise ValueError ('Given Qubit Operator can not be tapered')
      if term[-1][0] == alpha_parity_posn:
        if term[-1][1] == 'Z':
          term = term[:-1]
          coeff *= alpha_coeff
        else:
          raise ValueError ('Given Qubit Operator can not be tapered')
    JW_OF_tapered += QubitOperator(term, coeff)
  return JW_OF_tapered










def get_tapered_occ_str(occ_str):
  """
  Given a full space binary representation of a slater determinant, return the action of tapering unitary on the
  slater determinant. The result is encoded as a dictionary with keys consisting of binary strings representing
  slater determinants in the tapered space with values consisting of corresponding coefficients.

  Args:
      occ_str(str): Slater determinant as a string.

  Return:
      dict: Dictionary with keys representing slater deteriants and values representing their coefficients.
  """
  n_qubits = len(occ_str)
  U, U_dag = gen_tap_unitary(n_qubits)
  pauli_xs = occ_str_to_pauli_x(occ_str)
  rot_pauli_xs = U*pauli_xs*U_dag
  tobe_tapred = act_Q_OP_on_vacuum(rot_pauli_xs, n_qubits)
  tapered_dict = {}
  for key, val in tobe_tapred.items():
    tapered_dict[key[:-2]] = val
  return tapered_dict








def get_tapered_CI_states(ref_det, ranks, preserve_Sz = False):
  """
  Given a reference slater determinant and a set of excitation ranks, generates all CI type excitations,
  taper them, and return a sparse array whose columns are tapered slater determinants expressed
  in the Fock space basis of tapered space.

  Args:
      ref_det (str): Binary representation of the slater determiant whose excitations are to be found.
      ranks (list): List of non-negative integers corresponding to the rank of excitations. For example, if ranks = [0,1,4],
                    excitations include zeroth, first and fourth orders wrt reference state.
      preserve_Sz (optional, bool): It true, the excitations produced preserve the Sz spin of the reference determinant.

  Returns:
      scipy.sparse.csc_matrix: Sparse array whose columns are the tapered slater determinants in the tapered space.
  """
  alpha_parity = sum(int(i) for i in ref_det[::2])%2
  beta_parity = sum(int(i) for i in ref_det[1::2])%2

  n_qubits = len(list(ref_det))
  all_excitations = []
  if preserve_Sz == True:
    for r in ranks:
      all_excitations += get_Sz_preserving_nth_excitations(ref_det, r)
  else:
    for r in ranks:
      new_excitations = get_nth_excitations(ref_det, r)
      correct_excitations = []
      for new_ex in new_excitations:
        if (sum(int(i) for i in new_ex[::2])%2 == alpha_parity) and (sum(int(i) for i in new_ex[1::2])%2 == beta_parity):
          correct_excitations.append(new_ex)
      all_excitations += correct_excitations
  tap_CI_sarray = sp.sparse.csc_matrix((2**(n_qubits-2), len(all_excitations)), dtype='complex128')
  for ex_idx in range (len(all_excitations)):
    ex = all_excitations[ex_idx]

    new_str = get_tapered_occ_str(ex)

    vec = sp.sparse.csc_matrix((2**(n_qubits-2), 1), dtype='complex128')
    for string, coeff in new_str.items():
      vec += coeff*occ_str_to_state(string)
    tap_CI_sarray[:,[ex_idx]] = vec
  return tap_CI_sarray










def eval_tap_GS_CISD_overlap(JW_OF, n_qubits, num_elecs, verify_Sz_N_Ssq_evals = True):
  '''
  Given full space qubit hamiltonian, return the overlap of the ground state in the tapered space with the CISD approximation to the ground state.
  Also can verify if Sz = 0, N = num_elecs and S^2 = 0 for the CISD ground state. Assumes, num_alphas = num_betas.

  Args:
      JW_OF (QubitOperator): Input Hamiltonian as QubitOperator
      n_qubits (int): Number of qubits
      num_elecs (int): Number of electrons
      verify_Sz_N_Ssq_evals (bool, optional): If True, checks if Sz = 0, N = num_elecs and S^2 = 0
                                              for the CISD ground state. Defaults to True.

  Returns:
      float: Overlap of the ground state in tapered space with the CISD ground state.
  '''
  ref_det = '1'*num_elecs + '0'*(n_qubits-num_elecs)
  tapered_JW_OF = taper_qubits(JW_OF, n_qubits, int(num_elecs/2), int(num_elecs/2))
  tapered_JW_OF_sarray = get_sparse_operator(tapered_JW_OF, n_qubits-2)
  tapered_CI_states = get_tapered_CI_states(ref_det, [0,1,2], preserve_Sz=False)
  CIproj_tap_JW_OF_sarray = tapered_CI_states.T*tapered_JW_OF_sarray*tapered_CI_states

  CISD_v0, CISD_w0 = eigsh(np.real(CIproj_tap_JW_OF_sarray), k = 1, which='SA', return_eigenvectors=True)
  CISD_w0_sarray = sp.sparse.csc_matrix(CISD_w0)
  CISD_w0_full_space_sarray = tapered_CI_states*CISD_w0_sarray

  v0, w0_full_space = eigsh(tapered_JW_OF_sarray, k = 1, which='SA', return_eigenvectors=True)
  w0_full_space_sarray = sp.sparse.csc_matrix(w0_full_space)

  if verify_Sz_N_Ssq_evals == True:
    Ssq = s_squared_operator(n_qubits//2)
    tapred_Ssq = taper_qubits(jordan_wigner(Ssq), n_qubits, int(num_elecs/2), int(num_elecs/2))
    tapred_Ssq_sarray = get_sparse_operator(tapred_Ssq, n_qubits-2)
    Ssq_expval = np.abs(CISD_w0_full_space_sarray.T*tapred_Ssq_sarray*CISD_w0_full_space_sarray)[0,0]
    if np.abs(Ssq_expval) > 1e-6:
      print ('S^2 is not zero')
    else:
      print ('S^2 is zero')

    Sz = sz_operator(n_qubits//2)
    tapred_Sz = taper_qubits(jordan_wigner(Sz), n_qubits, int(num_elecs/2), int(num_elecs/2))
    tapred_Sz_sarray = get_sparse_operator(tapred_Sz, n_qubits-2)
    Sz_expval = np.abs(CISD_w0_full_space_sarray.T*tapred_Sz_sarray*CISD_w0_full_space_sarray)[0,0]
    if np.abs(Sz_expval) > 1e-6:
      print ('Sz is not zero')
    else:
      print ('Sz is zero')

    N_op = number_operator(n_qubits)
    tapred_N_op = taper_qubits(jordan_wigner(N_op), n_qubits, int(num_elecs/2), int(num_elecs/2))
    tapred_N_op_sarray = get_sparse_operator(tapred_N_op, n_qubits-2)
    N_expval = np.abs(CISD_w0_full_space_sarray.T*tapred_N_op_sarray*CISD_w0_full_space_sarray)[0,0]
    if np.abs(N_expval - num_elecs) > 1e-6:
      print ('N is not equal to num_elecs')
    else:
      print ('N is equal to num_elecs')

  overlap = (CISD_w0_full_space_sarray.T*w0_full_space_sarray)[0,0]
  return np.abs(overlap)**2
