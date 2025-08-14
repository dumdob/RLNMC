import sys

sys.path.append('.')

import jax
import jax.numpy as jnp
from jax import lax

from typing import Dict, Tuple, List, Union
from functools import partial, reduce

from modules.nmc_types import *

#Energy functions
@jax.jit
def Jxs_pubo(J: List[Jsp], 
						 h: jax.Array,
						 s: jax.Array) -> jax.Array:
	"""
	E = Q*x_1x_2...x_k... + ... +b*x for PUBO of any order
	s is a boolean vector: s = True/False = x = 1/0
	"""

	def term_prod(v, idx: jax.Array):
		return v*(jnp.take(s, idx, fill_value = False).all())
	v_term_prod = jax.vmap(term_prod)
	
	e = jnp.array(0, dtype = J[0].data.dtype)
	for j in J:
		e += v_term_prod(j.data, j.indices).sum()
	
	return e + h@s

@jax.jit
def Jxs_pising(J: List[Jsp], 
							 h: jax.Array,
							 s: jax.Array) -> jax.Array:
	"""
	E = J*s_1s_2...s_k + ... +h*s for p-spin Ising of any order
	s is a boolean vector: s = True/False = spin = 1/-1
	"""

	def term_prod(v, idx: jax.Array):
		return v*(1 - 2*reduce(jnp.logical_xor, jnp.logical_not(jnp.take(s, idx, fill_value = True))))
	v_term_prod = jax.vmap(term_prod)
	
	e = jnp.array(0, dtype = J[0].data.dtype)
	for j in J:
		e += v_term_prod(j.data, j.indices).sum()
	
	return e + h@(2*s-1)

@jax.jit
def Jxs_cnf(J: List[jax.Array],
						h: jax.Array,
        		s: jax.Array) -> jax.Array:
	"""
	cnf-based computation of the energy (faster than PUBO for k-SAT)
	s is a boolean vector: s = True/False = x = 1/0
	"""
	
	def cnf_prod(idx: jax.Array):
		return jnp.logical_xor(jnp.take(s, jnp.abs(idx) - 1, fill_value = False), 
												   jnp.signbit(idx)).all()
	v_cnf_prod = jax.vmap(cnf_prod)
	
	e = jnp.array(0, dtype = jnp.int32)
	for j in J: 
		e += v_cnf_prod(j).sum()
	
	return -e - jnp.abs(h)@jnp.logical_xor(s, jnp.logical_not(jnp.signbit(h)))

@jax.jit
def Jxs_wcnf(J: List[Jsp],
						 h: jax.Array,
        		 s: jax.Array) -> jax.Array:
	"""
	weigthed cnf-based computation of the energy (faster than PUBO for k-SAT)
	s is a boolean vector: s = True/False = x = 1/0
	"""

	def cnf_prod(v, idx: jax.Array):
		return v*jnp.logical_xor(jnp.take(s, jnp.abs(idx) - 1, fill_value = False), 
												   	jnp.signbit(idx)).all()
	v_cnf_prod = jax.vmap(cnf_prod)
	
	e = jnp.array(0, dtype = h.dtype)
	for j in J:
		e += v_cnf_prod(j.data, j.indices).sum()
	
	return -e - jnp.abs(h)@(jnp.logical_xor(s, jnp.logical_not(jnp.signbit(h))))

#Gradient functions
@jax.jit
def Jxs_de_pubo(Jat: List[Jsp], 
								s: jax.Array) -> jax.Array:
	"""
	dE = J*x_1x_2...dx_k + ... for PUBO of any order
	s is a boolean vector: s = True/False = x = 1/0
	"""

	def term_prod(v, idx: jax.Array):
		return v*(jnp.take(s, idx, fill_value = False).all())
	v_term_prod = jax.vmap(term_prod)
	
	de = jnp.array(0, dtype = Jat[0].data.dtype)
	for jat in Jat:
		de += v_term_prod(jat.data, jat.indices).sum()
	
	return de

@jax.jit
def Jxs_de_pising(Jat: List[Jsp], 
									s: jax.Array) -> jax.Array:
	"""
	dE = J*s_1s_2...ds_k + ... for p-spin Ising of any order
	s is a boolean vector: s = True/False = spin = 1/-1
	"""

	def term_prod(v, idx: jax.Array):
		return v*(1 - 2*reduce(jnp.logical_xor, jnp.logical_not(jnp.take(s, idx, fill_value = True))))
	v_term_prod = jax.vmap(term_prod)
	
	de = jnp.array(0, dtype = Jat[0].data.dtype)
	for jat in Jat:
		de += v_term_prod(jat.data, jat.indices).sum()
	
	return de


@jax.jit
def Jxs_de_cnf(Jat: List[jax.Array], 
        			 s: jax.Array) -> jax.Array:
	"""
	gain (gradient) of a variable = make - break for cnf (faster than PUBO for k-SAT)
	s is a boolean vector: s = True/False = x = 1/0
	"""

	def de_cnf_prod(idx: jax.Array):
		return idx[-1]*(jnp.logical_xor(jnp.take(s, jnp.abs(idx[0:-1]) - 1, fill_value = False), 
																		jnp.signbit(idx[0:-1])).all())
	v_de_cnf_prod = jax.vmap(de_cnf_prod)

	de = jnp.array(0, dtype = jnp.int32)
	for jat in Jat:
		de += v_de_cnf_prod(jat).sum() #parallelized make - break
	
	return de


@jax.jit
def Jxs_de_wcnf(Jat: List[Jsp], 
        			  s: jax.Array) -> jax.Array:
	"""
	gain (gradient) of a variable = make - break for weighted cnf (faster than PUBO for weighted k-SAT)
	s is a boolean vector: s = True/False = x = 1/0
	"""
	
	def de_cnf_prod(v, idx: jax.Array):
		return v*idx[-1]*(jnp.logical_xor(jnp.take(s, jnp.abs(idx[0:-1]) - 1, fill_value = False), 
																		jnp.signbit(idx[0:-1])).all())
	v_de_cnf_prod = jax.vmap(de_cnf_prod)
	
	de = jnp.array(0, dtype = Jat[0].data.dtype)
	for jat in Jat:
		de += v_de_cnf_prod(jat.data, jat.indices).sum() #parallelized make - break
	
	return de

@partial(jax.jit, static_argnames = ["pmode_idx"])
def	get_de(Jati: List[Jsp], 
					 hi: jax.Array,
        	 si: jax.Array,
					 s: jax.Array,
					 pmode_idx: int) -> jax.Array:

	if pmode_idx == 0: #PUBO
		de = (Jxs_de_pubo(Jati, s) + hi)*(2*si - 1)

	elif pmode_idx == 1: #p-spin Ising
		de = (Jxs_de_pising(Jati, s) + hi)*(4*si - 2)

	elif pmode_idx == 2: #cnf
		de = (Jxs_de_cnf(Jati, s) + hi)*(2*si - 1)

	else: #weighted cnf
		de = (Jxs_de_wcnf(Jati, s) + hi)*(2*si - 1)

	return de

@partial(jax.jit, static_argnames = ["pmode_idx"])
def get_energy(Jh: Jhdata,
							 s: jax.Array, 
							 pmode_idx: int) -> jax.Array:

	if pmode_idx == 0: #PUBO
		e = -Jxs_pubo(Jh.J, Jh.h, s)

	elif pmode_idx == 1: #p-spin Ising
		e = -Jxs_pising(Jh.J, Jh.h, s)

	elif pmode_idx == 2: #cnf
		e = -Jxs_cnf(Jh.J, Jh.h, s)

	else: #weighted cnf
		e = -Jxs_wcnf(Jh.J, Jh.h, s)

	return e

#####


#####
# NMC/MCMC sweeps logic
#####

@partial(jax.jit, static_argnames=['pmode_idx'])
def mc_step(Jh: Union[Jhdata, cnfdata],
						se: Sminsnokey,
						i,
						log_r,
						T: jax.Array,
						pmode_idx: int) -> Tuple[Sminsnokey, jax.Array]:
	"""
	Monte Carlo step at index i
	If T <= 0, then such spin is frozen
	"""
	s = se.s
	e = se.e

	if pmode_idx == 0:
		de = (Jxs_de_pubo([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(2*s[i] - 1)
	elif pmode_idx == 1:
		de = (Jxs_de_pising([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(4*s[i] - 2)
	elif pmode_idx == 2:
		de = (Jxs_de_cnf([jat[i] for jat in Jh.Jat], s) + Jh.h[i])*(2*s[i] - 1)
	else:
		de = (Jxs_de_wcnf([Jsp(jat.data[i], jat.indices[i]) for jat in Jh.Jat], s) + Jh.h[i])*(2*s[i] - 1)
	
	s, stat = lax.cond(jnp.logical_and(T > 0, log_r < -de/T), 
										lambda: (s.at[i].set(jnp.logical_not(s[i])), jnp.array([1, de])), 
										lambda: (s, jnp.array([0, 0], dtype=de.dtype)))

	# stat: (x, de), where x = 1 is flipped, 0 is not flipped, -1 is frozen
	e = lax.cond(stat[0] == 1, 
							lambda: e + de, 
							lambda: e)

	min_e, min_s = lax.cond(e < se.min_e,
													lambda: (e, s), 
													lambda: (se.min_e, se.min_s))
	
	return Sminsnokey(s, min_s, e, min_e), stat

@partial(jax.jit, static_argnames=['pmode_idx'])
def frozen_step(Jh: Union[Jhdata, cnfdata],
							  se: Sminsnokey,
							  i,
								log_r,
							  T: jax.Array,
							  pmode_idx: int) -> Tuple[Sminsnokey, jax.Array]:
	
	# stat: (x, de), where x = 1 is flipped, 0 is not flipped, -1 is frozen
	stat = jnp.array([-1, 0], se.e.dtype)
	return se, stat


@partial(jax.jit, static_argnames=['pmode_idx', 'Nsw', 'totrack', 'skip_frozen'])
def nmc_sampling(Jh: Union[Jhdata, cnfdata],
								 
							   sekey: Sminskey,

							   T: jax.Array,
								 
							   Nsw: int,
								 
								 skip_frozen: bool = True,
							   pmode_idx: int = 0, 
								 totrack: bool = True) -> Tuple[Sminskey, Dict]:
	"""
	NMC/MCMC sampling p-spin Ising/PUBO/w-cnf subroutine
	T controlls the fixed (T <= 0)/sampled (T > 0) subgraphs
	"""
	s_old = sekey.s
	e_old = sekey.e

	N = sekey.s.size #problem size

	def mc_cond_step_call(se: Sminsnokey, i_log_r):
		"""
		When no parallelization, not every step is performed based on T
		"""
		i = i_log_r[0]
		log_r = i_log_r[1]
		return lax.cond(T[i] > 0, 
										lambda: mc_step(Jh, se, i, log_r, T[i], pmode_idx), 
										lambda: frozen_step(Jh, se, i, log_r, T[i], pmode_idx))

	def mc_step_call(se: Sminsnokey, i_log_r):
		"""
		When parallelized, every step is performed regardless of T
		"""
		i = i_log_r[0]
		log_r = i_log_r[1]
		return mc_step(Jh, se, i, log_r, T[i], pmode_idx)
	
	def mc_sweep_call(sekey: Sminskey, 
									 	sweep_i: None):
		"""
		Monte Carlo seep function
		"""
		key, subkey1, subkey2 = jax.random.split(sekey.key, 3)
		
		perm = jax.random.permutation(subkey1, N) #generate permutation of the sweep
		log_r = jnp.log(jax.random.uniform(subkey2, (N,))) #generate random numbers for the sweep all at once
		
		se = Sminsnokey(sekey.s, sekey.min_s, sekey.e, sekey.min_e)

		if skip_frozen:
			se, stats = lax.scan(mc_cond_step_call, se, xs = (perm, log_r)) #type: ignore
		else:
			se, stats = lax.scan(mc_step_call, se, xs = (perm, log_r)) #type: ignore

		if totrack:
			sweep_accepted_flips = jnp.where(stats[:, 0] > 0, 1.0, 0.0).sum()
			sweep_total_flips = 	 jnp.where(stats[:, 0] >= 0, 1.0, 0.0).sum()

			sweep_accepted_climb_flips = jnp.where(stats[:, 1] > 0, 1.0, 0).sum()
			sweep_de_avg_climb = 	 			 jnp.where(stats[:, 1] > 0, stats[:, 1], 0).sum()

			sweep_stats = jnp.array([sweep_accepted_flips/(sweep_total_flips + 1e-9), 
															sweep_de_avg_climb/(sweep_accepted_climb_flips + 1e-9)])
		else:
			sweep_stats = jnp.empty(0)
		
		return Sminskey(se.s, se.min_s, se.e, se.min_e, key), sweep_stats

	#perform a for loop implementing Nsw MC sweeps
	sekey, sweep_stats_batch = lax.scan(mc_sweep_call, sekey, None, length = Nsw)
	
	#record the sampling statistics
	if totrack:
		sampling_stats = {
			"avg_ar": 	 sweep_stats_batch[:, 0].mean(),  					 # average acceptance rate of flips
			"avg_climb": sweep_stats_batch[:, 1].mean(),	 				 	 # average energy climb is flip is accepted with de > 0
			"avg_jump":	 (s_old != sekey.s).sum()/N, 								 # jump distance
			"avg_de":		 sekey.e - e_old 													 # energy change
		}
	else:
		sampling_stats = {}
	
	return sekey, sampling_stats


@partial(jax.jit, static_argnames=['pmode_idx', 'Nsw_eq', 'totrack'])
def sa_annealing(Jh: Union[Jhdata, cnfdata],
								 sekey: Sminskey,
								 beta_sa: jax.Array,
								 Nsw_eq: int,
								 pmode_idx: int = 0, 
								 totrack: bool = True) -> Tuple[Sminskey, Dict]:
	"""
	MCMC SA annealing (schedule of T) p-spin Ising/PUBO/w-cnf subroutine
	T controlls the fixed (T <= 0)/sampled (T > 0) subgraphs
	"""
	s_old = sekey.s
	e_old = sekey.e

	#linear schedule of the inverse temperature beta
	Tschedule = 1.0/jnp.linspace(beta_sa[0], beta_sa[1], Nsw_eq) 

	N = sekey.s.size #problem size
	
	def mc_sweep_call(sekey: Sminskey, temp):
		#MCMC sweep with a random permutation of flips
		
		def mc_step_call(se: Sminsnokey, i_logr):
			i = i_logr[0]
			log_r = i_logr[1]

			return mc_step(Jh, se, i, log_r, temp, pmode_idx)

		key, subkey1, subkey2 = jax.random.split(sekey.key, 3)

		perm = jax.random.permutation(subkey1, N) #generate permutation of the sweep
		log_r = jnp.log(jax.random.uniform(subkey2, (N,))) #generate random numbers for the sweep all at once
		
		se = Sminsnokey(sekey.s, sekey.min_s, sekey.e, sekey.min_e)

		se, stats = lax.scan(mc_step_call, se, xs = (perm, log_r)) #type: ignore
		
		if totrack:
			sweep_accepted_flips = jnp.where(stats[:, 0] > 0, 1.0, 0.0).sum()
			sweep_total_flips = 	 jnp.where(stats[:, 0] >= 0, 1.0, 0.0).sum()

			sweep_accepted_climb_flips = jnp.where(stats[:, 1] > 0, 1.0, 0).sum()
			sweep_de_avg_climb = 	 			 jnp.where(stats[:, 1] > 0, stats[:, 1], 0).sum()

			#sometimes the total number of flips at small temperature is zero
			sweep_stats = jnp.array([sweep_accepted_flips/(sweep_total_flips + 1e-9),
															 sweep_de_avg_climb/(sweep_accepted_climb_flips + 1e-9)])
		else:
			sweep_stats = jnp.empty(0)
		
		return Sminskey(se.s, se.min_s, se.e, se.min_e, key), sweep_stats

	#perform a for loop implementing Nsw MC sweeps
	sekey, sweep_stats_batch = lax.scan(mc_sweep_call, sekey, xs = Tschedule)
	
	#record the sampling statistics
	if totrack:
		sampling_stats = {
			"avg_ar": 	 sweep_stats_batch[:, 0].mean(),  					 # average acceptance rate of flips
			"avg_climb": sweep_stats_batch[:, 1].mean(),	 				 	 # average energy climb is flips accepted with de > 0
			"avg_jump":	 (s_old != sekey.s).sum()/N, 								 # jump distance
			"avg_de":		 sekey.e - e_old									 # energy change
		}
	else:
		sampling_stats = {}
	
	return sekey, sampling_stats

@partial(jax.jit, static_argnames=['pmode_idx', 'totrack'])
def nmc_randomize(Jh: Union[Jhdata, cnfdata],
									sekey: Sminskey,
									T: jax.Array,
									pmode_idx: int = 0,
									totrack: bool = True) -> Tuple[Sminskey, Dict]:
	"""
	Backbone randomization of spins
	"""
	s_old = sekey.s
	e_old = sekey.e

	N = sekey.s.size #problem size

	key, subkey = jax.random.split(sekey.key)
	s_rnd = jax.random.choice(subkey, jnp.array([False, True]), shape = (N,))
	
	s =	jnp.where(T > 0, s_rnd, sekey.s)
	e = get_energy(Jh, s, pmode_idx)

	min_s, min_e = lax.cond(e < sekey.min_e, 
													lambda: (s, e), 
													lambda: (sekey.min_s, sekey.min_e))
	

	if totrack:
		sampling_stats = {
			"avg_ar": 	 1.0,  					 					# average acceptance rate of flips
			"avg_climb": (e - e_old).mean(),	 		# average energy climb is flip is accepted with de > 0
			"avg_jump":	 (s_old != s).sum()/N, 		# jump distance
			"avg_de":		 e - e_old			# energy change
		}
	else:
		sampling_stats = {}

	return Sminskey(s, min_s, e, min_e, key), sampling_stats

#####		

@partial(jax.jit, static_argnames=['Ncycles', 'Nsw_bb', 'Nsw_nbb', 'Nsw_eq', 'skip_frozen', 
																	 'eq_annealing', 'pmode_idx', 'totrack', 'rnd_jump'])
def nmc_cycles(Jh: Union[Jhdata, cnfdata],
							 minskey: Minskey,

							 best_s: jax.Array,
							 best_e: jax.Array,

							 xT: jax.Array,
							 T: jax.Array,
							 Tf: jax.Array,
							
							 Ncycles: int,
							 Nsw_bb: int,
							 Nsw_nbb: int,
							 Nsw_eq: int,
							
							 eq_annealing: bool = False,
							 rnd_jump: bool = False,
							 skip_frozen: bool = True,
							 pmode_idx: int = 0,
							 totrack: bool = True
							 ) -> Tuple[Minskey, 
													jax.Array, 
													jax.Array, 
													Dict]:			 			 
	"""
	NMC cycles which include in/out of backbone sampling/randomization + equilibrium sampling
	"""

	#xT multipler for the backbone phase
	Tb = jnp.array(jnp.where(xT > 1, xT*T, -1), dtype=jnp.float32)
	
	#x1 T multipler for the non-backbone phase
	Tnb = jnp.array(jnp.where(xT > 1, -1, T), dtype=jnp.float32)

	if eq_annealing:
		# Initial-final temperatures for the equilibrium annealing phase
		beta_sa = jnp.array([1.0/T, 1.0/Tf], dtype=jnp.float32)
	else:
		# T for the full phase without annealing
		Teq = jnp.full_like(xT, T, dtype=jnp.float32)

	#save the minimum energy and state from the previous basin
	#the Ncycles number of NMC jumps will be performed from this state
	old_s = minskey.min_s
	old_e = minskey.min_e

	if old_e.dtype == jnp.int32:
		infty = jnp.full_like(old_e, jnp.iinfo(jnp.int32).max)
	elif old_e.dtype == jnp.float32:
		infty = jnp.full_like(old_e, jnp.inf)
	else:
		raise RuntimeError("Wrong dtype")
	
	def NMC_cycles(minskey: Minskey, i: None):
		sminskey = lax.cond(i == 0, lambda: Sminskey(old_s, minskey.min_s, old_e, infty, minskey.key), #reset the first min_e
																lambda: Sminskey(old_s, minskey.min_s, old_e, minskey.min_e, minskey.key))

		#backbone sweep
		if rnd_jump:
			sminskey, stats_bb = nmc_randomize(Jh, sminskey, Tb, pmode_idx, totrack)
		else:
			sminskey, stats_bb = nmc_sampling(Jh, sminskey, Tb, Nsw_bb, skip_frozen, pmode_idx, totrack)

		#non-backbone sweep
		if rnd_jump:
			sminskey, stats_nbb = nmc_sampling(Jh, sminskey, Tnb, 1, skip_frozen, pmode_idx, totrack)
		else:
			sminskey, stats_nbb = nmc_sampling(Jh, sminskey, Tnb, Nsw_nbb, skip_frozen, pmode_idx, totrack)

		#equilibrium sweep
		if eq_annealing:
			sminskey, stats_eq = sa_annealing(Jh, sminskey, beta_sa, Nsw_eq, pmode_idx, totrack)
		else:
			sminskey, stats_eq = nmc_sampling(Jh, sminskey, Teq, Nsw_eq, skip_frozen, pmode_idx, totrack)

		minskey = Minskey(sminskey.min_s, sminskey.min_e, sminskey.key)
		return (minskey, (stats_bb, stats_nbb, stats_eq))

	#the best state from Ncycles states is stored
	minskey, nmc_cycle_stats = lax.scan(NMC_cycles, minskey, xs = jnp.arange(Ncycles))

	if totrack:
		mc_stats = {"bb": {}, "nbb": {}, "eq": {}}
		for dkey in nmc_cycle_stats[0].keys():
			mc_stats["bb"][dkey]  = nmc_cycle_stats[0][dkey].mean()
			mc_stats["nbb"][dkey] = nmc_cycle_stats[1][dkey].mean()
			mc_stats["eq"][dkey]  = nmc_cycle_stats[2][dkey].mean()
	else:
		mc_stats = {}

	mc_stats["min_jump"] = (old_s != minskey.min_s).sum()/minskey.min_s.size		#total jump betweeen minima distance
	mc_stats["min_de"] = minskey.min_e - old_e								  								#total energy change between minima

	best_s, best_e = lax.cond(minskey.min_e < best_e,
														lambda: (minskey.min_s, minskey.min_e), 
														lambda: (best_s, best_e))
														
	return minskey, best_s, best_e, mc_stats
