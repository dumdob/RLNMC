from os import path 

import sys
sys.path.append('.')

import numpy as np
import torch as th
import jax.numpy as jnp
from jax.experimental import sparse
from modules.nmc_types import *

from typing import Tuple, List, Dict, Union, Any

def unroll_clause_to_polynomial(monomial: Tuple[Any, List[Tuple[str, int]]], 
															 polynomial: Dict, constant: List[Any], pmode):
	"""
	Helper function that transforms the cnf formulation of the clause
	to either p-Ising or PUBO
	"""

	if (monomial[1] == [('s', abs(m[1])) for m in monomial[1]] and pmode == 'pising'): 
		polynomial[len(monomial[1])]["indices"].append([m[1]-1 for m in monomial[1]])
		polynomial[len(monomial[1])]["values"].append(monomial[0])

	elif (monomial[1] == [('b', abs(m[1])) for m in monomial[1]] and pmode == 'pubo'): 

		polynomial[len(monomial[1])]["indices"].append([m[1]-1 for m in monomial[1]])
		polynomial[len(monomial[1])]["values"].append(monomial[0])
	 
	elif(monomial[1][0][0] == 'c'):
		constant[0] += 1
	 
	else:
		for i, m in enumerate(monomial[1]):
			if m[0] == 'x':
				if pmode == 'pising':
					new_monomial_a = (monomial[0]*np.sign(m[1]), monomial[1].copy())
					new_monomial_a[1][i] = ('s', abs(m[1]))
					
					new_monomial_b = (monomial[0], [])
					new_monomial_b[1].extend(monomial[1][:i])

					if i != len(monomial[1])-1:
						new_monomial_b[1].extend(monomial[1][i+1:])
					
					if len(new_monomial_b[1]) == 0:
						new_monomial_b[1].append(('c', 0))
					
					unroll_clause_to_polynomial(new_monomial_a, polynomial, constant, pmode)
					unroll_clause_to_polynomial(new_monomial_b, polynomial, constant, pmode)

				elif pmode == 'pubo':
					new_monomial_a = (monomial[0]*np.sign(m[1]), monomial[1].copy())
					new_monomial_a[1][i] = ('b', abs(m[1]))

					unroll_clause_to_polynomial(new_monomial_a, polynomial, constant, pmode)

					if np.sign(m[1]) < 0:
						new_monomial_b = (monomial[0], [])
						new_monomial_b[1].extend(monomial[1][:i])

						if i != len(monomial[1])-1:
							new_monomial_b[1].extend(monomial[1][i+1:])
						
						if len(new_monomial_b[1]) == 0:
							new_monomial_b[1].append(('c', 0))
					
						unroll_clause_to_polynomial(new_monomial_b, polynomial, constant, pmode)

				else:
					raise RuntimeError("Wrong pmode!")

				break

# (1-xa)(1-xb)(xc)(1-xd) = (1-sa)(1-sb)(1+s)(1-sd)/16
# x = (1+s)/2

def wcnf_to_sparse_p(directory: str, 
										instance_name: str,
										model_dtype: Any,
										N: int = 0, 
										sign: int = 1,
										pmode: Union[str, None] = "pising",
										no_weights: bool = False,
										gs_file = None) -> Tuple[Tuple[List[Any], th.Tensor], Any, Any]:
	"""
	Read a wcnf file into sparse matrices J and dense vector h of the p-Ising/PUBO formulation of energy.

	Parameters:
	- directory (str): Path to the cnf file
	- instance_name (str): instance_name.cnf
	- dtype: np.float32 or np.int32 type for the interaction matrix/bias
	- N (int): If needed to set the minimum problem size, else 0
	- sign (int): the sign of the p-Ising, PUBO formulation: +/- sum J_ij x_i x_j
	- pmode (str): pubo, cnf, or pising formulation
	- no_weights (bool): True if sat is not weighted

	Returns:
	- tuple: Tuple containing the tuple of matrices J and the vector h, the ground state (if known), and all clauses
	"""

	sJ = []

	ground_state = 0.0
	polynomial = {}

	dtype = np.int32 if model_dtype == "int" else np.float32
	all_clauses = {}

	maxK = 0

	with open(path.join(directory, instance_name), 'r') as f:
		for line in f:
			if line.startswith('c') or line.startswith('p'):
				continue

			L = np.fromstring(line, dtype = dtype, sep = ' ')
			if no_weights:
				indices = L[:-1].astype(int)
			else:
				indices = L[1:-1].astype(int)

			K = len(indices)

			if K not in all_clauses.keys():
				#indices and weights
				if no_weights:
					all_clauses[K] = np.empty((0, K), dtype = np.int32)
				else:
					all_clauses[K] = [np.empty((0,), dtype = dtype), 
											 			np.empty((0, K), dtype = np.int32)]

			if no_weights:
				all_clauses[K] = np.append(all_clauses[K], indices)

			else:
				all_clauses[K][0] = np.append(all_clauses[K][0], L[0]) #set the weight of the clause
				all_clauses[K][1] = np.append(all_clauses[K][1], indices)

			if pmode not in ["cnf", "wcnf"]:
				if K > maxK:
					maxK = K
				
				for k in range(1, K + 1):
					if k not in polynomial:
						polynomial.update({k: {"indices": [], "values": []}})
				
				
				lineN = np.max(np.abs(indices))
				if lineN > N:
					N = int(lineN)
				
				#weight multipler for monomials
				if no_weights:
					w = 1
				else:
					w = L[0]

				constant = [0] #modified in place
				if pmode =='pising':
					unroll_clause_to_polynomial((pow(2, -K)*w, [('x', -l) for l in indices]), 
																			polynomial, constant, pmode)

					ground_state += constant[0]*pow(2, -K)*w

				elif pmode == 'pubo':
					unroll_clause_to_polynomial((w, [('x', -l) for l in indices]), 
																				polynomial, constant, pmode)

					ground_state += constant[0]*w
				else: 
					raise RuntimeError("Wrong pmode")

	if pmode in ["cnf", "wcnf"]:
		return None, None, all_clauses

	dtype = th.int32 if model_dtype == "int" else th.float32
	h = th.zeros(N, dtype = dtype)
	for key, value in sorted(polynomial.items(), reverse=False):
		if key != 1:
			sp_tensor = th.sparse_coo_tensor(np.transpose(np.sort(np.array(value["indices"]), axis=1)), #type: ignore
																			 value["values"], 
																			 size = tuple(np.full(key, N))).coalesce()

			#to remove possible zero elements from the coalesce operation (is there a function to do it for tensors?)	 
			mask = sp_tensor.values().nonzero()
			nnzv = sp_tensor.values().index_select(0, mask.view(-1))
			nnzi = sp_tensor.indices().index_select(1, mask.view(-1))

			#create a coalesced tensor again		
			sp_tensor = th.sparse_coo_tensor(nnzi,
																			 nnzv*pow(2, maxK) if pmode == "pising" else nnzv, 
																			 size = tuple(np.full(key, N)), dtype = dtype).coalesce()
			
			sJ.append(-sp_tensor if sign > 0 else sp_tensor)

		else: 
			values = np.array(value["values"])*pow(2, maxK) if pmode == "pising" else value["values"]
			for i, v in zip(value["indices"], values):
				if sign > 0:
					h[i[0]] -= v
				else:
					h[i[0]] += v

	if sign > 0:
		ground_state = -ground_state
	
	if pmode == "pising":
		ground_state *= pow(2, maxK)

	return (sJ, h), ground_state, all_clauses


def sparse_ising_for_jax(instance: Tuple[List[Any], Union[th.Tensor, np.ndarray]]) -> Jhdata:
	J_bcoo = [sparse.BCOO((sJ.values().numpy(), sJ.indices().numpy().transpose()),
										 	shape=sJ.shape) 
					for sJ in instance[0]]
	h = jnp.array(instance[1])

	N = h.size
	Jat_all = [[] for _ in range(N)]
	
	max_nses = []
	for j, J in enumerate(J_bcoo):
		max_nses.append(0)

		K = J.ndim
		slice_array_start = np.full(K, 0)
		slice_array_end = np.full(K, N)

		for i in range(N):
			for k in range(K):
				sl_start = slice_array_start.copy()
				sl_end = slice_array_end.copy()

				sl_start[k] = i
				sl_end[k] = i+1

				Jtmp = sparse.bcoo_squeeze(
								sparse.bcoo_sum_duplicates(
									sparse.bcoo_slice(J,
																		start_indices = sl_start, #type: ignore
																		limit_indices = sl_end) #type: ignore
								), dimensions=[k]
							)

				if k == 0:
					Jat_all[i].append(Jtmp)
				else:
					Jat_all[i][j] += Jtmp

			if Jat_all[i][j].nse > max_nses[-1]:
				max_nses[-1] = Jat_all[i][j].nse

		for i in range(N):
			# do the padding of all sparse matrices to one (maximum) size so that nmc is jit compiled
			Jat_all[i][j] = sparse.bcoo_sum_duplicates(Jat_all[i][j], nse=max_nses[-1])
	
	Jat = []
	for j, J in enumerate(J_bcoo):
		K = J.ndim

		data_stacked = jnp.stack([Jat_all[i][j].data for i in range(N)])
		indices_stacked = jnp.stack([Jat_all[i][j].indices for i in range(N)])
		Jat.append(Jsp(data_stacked, indices_stacked))

	J_v = []
	for j_bcoo in J_bcoo:
		J_v.append(Jsp(j_bcoo.data, j_bcoo.indices))

	return Jhdata(J_v, h, Jat)


def cnf_for_jax(cnf_dict: Dict[int, np.ndarray]) -> cnfdata:
	J_v = []
	Jat = []

	N = 0

	for k, cnf in sorted(cnf_dict.items(), reverse=False):
		n = np.max(np.abs(cnf))
		if n > N:
			N = n
		if k > 1:
			J_v.append(-jnp.array(cnf, dtype=jnp.int32).reshape(-1, k))

	h = jnp.zeros(N, dtype=jnp.int32)

	for k, cnf in sorted(cnf_dict.items(), reverse=False):
		jat = [[] for _ in range(N)]

		if k == 1:
			for l in cnf:
				h = h.at[abs(l)-1].set(np.sign(l))
		else:
			for l in cnf.reshape(-1, k):
				for ci, c in enumerate(l):
					ltmp = -np.delete(l, ci)
					ltmp = np.append(ltmp, np.sign(c))
					jat[abs(c)-1].append(ltmp)

			maxlen = np.max([len(j) for j in jat])
			Jat.append(jnp.zeros((N, maxlen, k), dtype=jnp.int32))
			for i in range(N):
				if len(jat[i]) != 0:
					Jat[-1] = Jat[-1].at[i, :len(jat[i]), :].set(jnp.stack(jat[i]))

	return cnfdata(J_v, h, Jat)


def wcnf_for_jax(wcnf_dict: Dict[int, WCNF], dtype) -> Jhdata:
	J_v = []
	Jat = []

	N = 0

	for k, wcnf in sorted(wcnf_dict.items(), reverse=False):
		n = np.max(np.abs(wcnf.indices))
		if n > N:
			N = n
		if k > 1:
			J_v.append(Jsp(jnp.array(wcnf.w, dtype = dtype), 
									-jnp.array(wcnf.indices, dtype=jnp.int32).reshape(-1, k)))

	h = jnp.zeros(N, dtype = dtype)

	for k, wcnf in sorted(wcnf_dict.items(), reverse=False):
		jat_w = [[] for _ in range(N)]
		jat_idx = [[] for _ in range(N)]

		if k == 1:
			for i, l in enumerate(wcnf.indices):
				h = h.at[abs(l)-1].set(np.sign(l)*wcnf.w[i])
		else:
			for i, l in enumerate(wcnf.indices.reshape(-1, k)):
				for ci, c in enumerate(l):
					ltmp = -np.delete(l, ci)
					ltmp = np.append(ltmp, np.sign(c))

					jat_idx[abs(c)-1].append(ltmp)
					jat_w[abs(c)-1].append(wcnf.w[i])

			maxlen = np.max([len(j) for j in jat_idx])

			data = jnp.zeros((N, maxlen), dtype = dtype)
			indices = jnp.zeros((N, maxlen, k), dtype = jnp.int32)

			for i in range(N):
				if len(jat_idx[i]) != 0:
					indices = indices.at[i, :len(jat_idx[i]), :].set(jnp.stack(jat_idx[i]))
					data = data.at[i, :len(jat_w[i])].set(jnp.array(jat_w[i]))

			Jat.append(Jsp(data, indices))

	return Jhdata(J_v, h, Jat)

def pad_and_stack_Jh(Jh_list: List[Jhdata]) -> Jhdata:
	"""
	Pad the interaction matrices with zeros and stack them into a single Jhdata
	"""
	N = Jh_list[0].h.size
	assert (jnp.all(jnp.array([Jh.h.size == N for Jh in Jh_list]))), "Not all problems are of equal size"

	k_order = len(Jh_list[0].J)

	#max size of the sparse interaction matrices (J) of all orders
	J_sizes = [np.max([Jh.J[k].data.size for Jh in Jh_list]) for k in range(k_order)]

	#max size of the sparse interaction matrices for every index (Jat) of all orders 
	Jat_sizes = [np.max([Jh.Jat[k].data.shape[1] for Jh in Jh_list]) for k in range(k_order)]
	
	J = []
	h = jnp.stack([Jh.h for Jh in Jh_list])
	Jat = []
	for k in range(k_order):
		jsp = Jsp(jnp.stack([jax.lax.pad(Jh.J[k].data, jnp.array(0, dtype = Jh.J[k].data.dtype), 
																		 [(0, J_sizes[k] - Jh.J[k].data.shape[0], 0), ]) for Jh in Jh_list]),
							jnp.stack([jax.lax.pad(Jh.J[k].indices, 0, 
																		 [(0, J_sizes[k] - Jh.J[k].indices.shape[0], 0), (0, 0, 0)]) for Jh in Jh_list]))
		J.append(jsp)

		jat = Jsp(jnp.stack([jax.lax.pad(Jh.Jat[k].data, jnp.array(0, dtype = Jh.Jat[k].data.dtype), 
																		 [(0, 0, 0), (0, Jat_sizes[k] - Jh.Jat[k].data.shape[1], 0)]) for Jh in Jh_list]),
							jnp.stack([jax.lax.pad(Jh.Jat[k].indices, 0, 
																		 [(0, 0, 0), (0, Jat_sizes[k] - Jh.Jat[k].indices.shape[1], 0), (0, 0, 0)]) for Jh in Jh_list]))
		Jat.append(jat)


	return Jhdata(J, h, Jat)


def pad_and_concatenate_Jh(Jh_list: List[Jhdata], 
													 pmode_idx) -> Jhdata:
	"""
	Pad the interaction matrices with zeros and cat them into a single Jhdata (not stacking)
	"""
	N = Jh_list[0].h.size
	assert (np.all([Jh.h.size == N for Jh in Jh_list])), "Not all problems are of equal size"

	p_order = len(Jh_list[0].J)

	if pmode_idx in [0, 1]:
		J = [Jsp(jnp.concatenate([Jh.J[p].data for Jh in Jh_list], axis = 0),
								jnp.concatenate([Jh.J[p].indices + i*N for i, Jh in enumerate(Jh_list)], axis = 0)) 
				 for p in range(p_order)]
	else:
		J = [Jsp(jnp.concatenate([Jh.J[p].data for Jh in Jh_list], axis = 0),
						jnp.concatenate([Jh.J[p].indices + jnp.sign(Jh.J[p].indices)*i*N for i, Jh in enumerate(Jh_list)], axis = 0)) 
				for p in range(p_order)]

	#max size of the sparse interaction matrices for every index (Jat) of all orders 
	Jat_sizes = [np.max([Jh.Jat[p].data.shape[1] for Jh in Jh_list]) for p in range(p_order)]

	h = jnp.concatenate([Jh.h for Jh in Jh_list])
	
	Jat = []
	for p in range(p_order):
		jat_data = [jax.lax.pad(Jh.Jat[p].data, jnp.array(0, dtype = Jh.Jat[p].data.dtype), 
													  [(0, 0, 0), (0, Jat_sizes[p] - Jh.Jat[p].data.shape[1], 0)]) for Jh in Jh_list]
		jat_indices = [jax.lax.pad(Jh.Jat[p].indices, 0, 
															 [(0, 0, 0), (0, Jat_sizes[p] - Jh.Jat[p].indices.shape[1], 0), (0, 0, 0)]) for Jh in Jh_list]

		if pmode_idx in [0, 1]:
			Jat.append(Jsp(jnp.concatenate([jat_d for jat_d in jat_data], axis = 0), 
										 jnp.concatenate([jat_idx + i*N for i, jat_idx in enumerate(jat_indices)])))
		else:
			Jat.append(Jsp(jnp.concatenate([jat_d for jat_d in jat_data], axis = 0), 
										 jnp.concatenate([jat_idx + jnp.concatenate([jnp.sign(jat_idx[:, :, :-1])*i*N, 
																											 					 jnp.zeros((N, Jat_sizes[p], 1), dtype = int)], axis = 2)
														for i, jat_idx in enumerate(jat_indices)], axis = 0)))

	return Jhdata(J, h, Jat)

