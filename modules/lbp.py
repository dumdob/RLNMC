import sys
import itertools

sys.path.append('.')

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from typing import Dict, Tuple, List, Union
from functools import partial
from dataclasses import replace

from pgmax import fgraph
from pgmax import fgroup
from pgmax import vgroup
from pgmax import infer

from modules.nmc_types import *

def init_factor_graph(J: Union[List[Jsp], List[jax.Array]],
											h: jax.Array, 
											lbp_beta: float,
											pmode_idx: int = 0) -> fgraph.FactorGraph:
	N = h.size

	variables = vgroup.NDVarArray(num_states = 2, shape = (N,)) 
	factor_graph = fgraph.FactorGraph(variable_groups = variables)

	if pmode_idx == 0: #pubo biases
		h_potentials = np.stack([h, np.zeros_like(h)], axis = 1).astype(np.float32)*lbp_beta # E = -b_ix_i

	elif pmode_idx == 1: #pising magnetic fields
		h_potentials = np.stack([h, -h], axis = 1).astype(np.float32)*lbp_beta # E = -h_is_i

	else: #cnf weighted single clauses
		h_potentials = np.stack([h, np.zeros_like(h)], axis = 1).astype(np.float32)*lbp_beta # E = w_ix_i (h < 0) or w_i(1-x_i)~-w_ix_i (h > 0)

	h_factors = fgroup.EnumFactorGroup(
    variables_for_factors = [[variables[i]] for i in range(N)],
    factor_configs = (np.array([1, 0]))[:, None],
    log_potentials = h_potentials
	)

	J_factors = []
	for j in J:
		if isinstance(j, Jsp):
			data = j.data
			indices = j.indices
		else:
			indices = j
			data = jnp.full((indices.shape[0],), 1.0)

		k = indices.shape[1]
		if pmode_idx in [0, 1]:
			variables_for_factors = [[variables[i] for i in idx] for idx in indices]
		else:
			#in cnf indexing in clauses is with a sign and starts from 1
			variables_for_factors = [[variables[np.abs(i)-1] for i in idx] for idx in indices]

		factor_configs = np.array(list(itertools.product([1, 0], repeat = k)))
		if pmode_idx == 0:
			log_potentials = data[:, None]*np.prod(factor_configs, axis=1).astype(np.float32)*lbp_beta

		elif pmode_idx == 1:
			s_configs = np.array(list(itertools.product([1, -1], repeat = k)))
			log_potentials = data[:, None]*np.prod(s_configs, axis=1).astype(np.float32)*lbp_beta

		else:
			#sort indices in each clause for possible duplicates removal
			sort_ind = np.argsort(np.abs(indices), axis = 1)
			indices = jnp.take_along_axis(indices, sort_ind, axis = 1)

			s_configs = np.array(list(itertools.product([1, -1], repeat = k)))
			clause_evaluations = np.prod(indices[:, None, :]*s_configs > 0, axis=2).astype(np.float32)
			log_potentials = -data[:, None]*clause_evaluations*lbp_beta

			#remove duplicates from the factor graph (merge factors with the same variables)
			indices_unique = np.unique(np.abs(indices), axis = 0, return_counts = True)
			indices_unique_where = np.where(indices_unique[1] > 1)[0]
			if indices_unique_where.size > 0:
				indices_unique = indices_unique[0][indices_unique_where]
				for idx in indices_unique:
					where_duplicates = np.where((idx == np.abs(indices)).all(axis = 1))[0]
					log_potential_sum = np.sum(log_potentials[where_duplicates], axis = 0)
					indices = np.stack([x for i, x in enumerate(indices) if i not in where_duplicates] + [indices[where_duplicates[0]]])
					variables_for_factors = (
						[x for i, x in enumerate(variables_for_factors) if i not in where_duplicates] + 
						[variables_for_factors[where_duplicates[0]]]
					)
					log_potentials = np.stack([x for i, x in enumerate(log_potentials) if i not in where_duplicates] + [log_potential_sum])
		
		J_factors.append(fgroup.EnumFactorGroup(variables_for_factors, factor_configs, log_potentials))

	factor_graph.add_factors([h_factors] + J_factors)

	return factor_graph

@partial(jax.jit, static_argnames=['bp', 'FGvars', 'num_iters'])
def surrogate_LBP(bp,
									FGvars,
									s: jax.Array, 
									epsilon: jax.Array,
									lbp_beta: float,
									num_iters: int,
									tolerance_m: float,
									tolerance_d: float,
									lambdas: jax.Array) ->  Tuple[jax.Array, Dict]:
	"""
	Loopy belief propagation on the problem's factor graph and using the convexified surrogate Hamiltonian

	Parameters:
		s (jax.Array): current state of the problem
		epsilon (jax.Array): A scaling factor for every spin to account for different interactions
		lbp_beta (float): temperature of lbp
		num_iters (int): pgmax lbp number of iterations
		tolerance_m (float): a maximum allowed bp relative diff of messages to stop
		tolerance_d (float): a maximum allowed distance from the initial state to stop
		lambdas (Tuple): control parameter of the "pinned" Loopy Belief Propagation
	"""
	run_with_diffs = partial(bp.run_with_diffs, #type: ignore
													 num_iters = num_iters, 
													 damping = 0.0, 
													 temperature = 1.0)
	
	def map_s_distance(m, s):
		return (m != s.astype(int)).sum()/s.size
	
	evidence_0 = (jnp.stack([1-2*s, 2*s-1], axis=1)*epsilon[:, None]).astype(jnp.float32)*lbp_beta

	bp_arrays = bp.init(evidence_updates = {FGvars: evidence_0*lambdas[0]})
	bp_arrays, msgs_eps = run_with_diffs(bp_arrays)
	map_state = infer.decode_map_states(bp.get_beliefs(bp_arrays))[FGvars]
	
	ms_d = map_s_distance(map_state, s)

	lbp_stats = {}

	lbp_stats["start_msgs_delta"] = msgs_eps[-1]
	lbp_stats["start_distance"] =  ms_d
	lbp_stats["start_lambda"] =  lambdas[0]

	# jax.debug.print("Start messages delta: {x}", x = msgs_eps[-1])
	# jax.debug.print("Start distance: {x}", x = ms_d)
	# jax.debug.print("Start lambda: {x}",  x = lambdas[0])
	
	#Setting up initial lambda (increasing) in case LBP diverged beyond the threshold
	def run_bp_lmbd_increase(x):
		lmbd = x[3]/lambdas[2]

		bp_arrays = bp.init(evidence_updates = {FGvars: evidence_0*lmbd})

		bp_arrays, msgs_eps = run_with_diffs(bp_arrays)

		ms_d = map_s_distance(infer.decode_map_states(bp.get_beliefs(bp_arrays))[FGvars], s)

		# jax.debug.print("Setting up: dmsgs = {x}, ms_d = {y}, lambda = {z}", x = msgs_eps[-1], y = ms_d, z = lmbd)

		return bp_arrays, msgs_eps[-1], ms_d, lmbd, x[4] + 1
	
	def set_lambda_init(bp_arr, m_eps, ms_d, lmbd):
		return lax.while_loop(lambda x: jnp.logical_or(x[1] > tolerance_m, x[2] > tolerance_d), 
													run_bp_lmbd_increase, (bp_arr, m_eps, ms_d, lmbd, 0))

	bp_arrays, m_eps, ms_d, lmbd, init_steps = \
		lax.cond(jnp.logical_or(msgs_eps[-1] > tolerance_m, ms_d > tolerance_d), 
						 set_lambda_init, 
						 lambda bp_arr, m_eps, ms_d, lmbd: (bp_arr, m_eps, ms_d, lmbd, 0), 
						 bp_arrays, msgs_eps[-1], ms_d, lambdas[0])	


	lbp_stats["init_steps"] = init_steps #number of initialization steps
	lbp_stats["init_msgs_delta"] = m_eps
	lbp_stats["init_distance"] = ms_d
	lbp_stats["init_lambda"] = lmbd

	# jax.debug.print("Init messages delta: {x}", x = m_eps)
	# jax.debug.print("Init distance: {x}", x = ms_d)
	# jax.debug.print("Init lambda: {x}",  x = lmbd)


	#Reducing lambda until LBP diverges beyond the threshold
	def run_bp_lmbd_decrease(x):
		bp_arrays_old = x[0][0]
		m_delta_old = x[0][1]
		ms_d_old = x[0][2]
		lmbd_old = x[0][3]

		lmbd = x[0][3]*lambdas[2] #decrease the value of lambda
		#initialize bp messages with the previous iteration and updated evidence
		bp_arrays = replace(x[0][0], evidence = evidence_0.flatten()*lmbd) 

		bp_arrays, msgs_eps = run_with_diffs(bp_arrays)
		ms_d = map_s_distance(infer.decode_map_states(bp.get_beliefs(bp_arrays))[FGvars], s)

		# jax.debug.print("Adjusting: dmsgs = {x}, ms_d = {y}, lambda = {z}", x = msgs_eps[-1], y = ms_d, z = lmbd)

		return (bp_arrays, msgs_eps[-1], ms_d, lmbd), (bp_arrays_old, m_delta_old, ms_d_old, lmbd_old), x[2] + 1
	

	final_lbp_res, pre_final_lbp_res, final_steps  = \
		lax.while_loop(lambda x: jnp.logical_and(jnp.logical_and(x[0][1] < tolerance_m, x[0][2] < tolerance_d), x[0][3] > lambdas[1]), 
									 run_bp_lmbd_decrease, 
									 ((bp_arrays, m_eps, ms_d, lmbd), (bp_arrays, m_eps, ms_d, lmbd), 0))


	lbp_res = lax.cond(jnp.logical_or(final_lbp_res[1] > tolerance_m, final_lbp_res[2] > tolerance_d), 
										 lambda: pre_final_lbp_res, 
										 lambda: final_lbp_res)

	lbp_stats["final_steps"] = final_steps #final number of lbp steps
	lbp_stats["final_msgs_delta"] = lbp_res[1]
	lbp_stats["final_distance"] = lbp_res[2]
	lbp_stats["final_lambda"] = lbp_res[3]

	# jax.debug.print("Final messages delta: {x}", x = lbp_res[1])
	# jax.debug.print("Final distance: {x}", x = lbp_res[2])
	# jax.debug.print("Final lambda: {x}",  x = lbp_res[3])

	beliefs = bp.get_beliefs(lbp_res[0])

	#return the final beliefs of the LBP convergence and the adjusted starting lambda value of LBP
	return beliefs, lbp_stats 
