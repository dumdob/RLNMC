import sys
sys.path.append('.')

NO_LBP = True

import jax
import jax.numpy as jnp
from jax import lax

from flax import struct

from gymnax.environments import environment
from gymnax.environments import spaces

from modules import nmc
from modules import lbp

from pgmax import infer

from functools import partial

from typing import Any, Dict, Tuple, Union, Optional

@struct.dataclass
class EnvObs:
	s: jax.Array
	r: jax.Array
	e: jax.Array
	best_e: jax.Array
	dtobest: jax.Array
	temp: jax.Array

@struct.dataclass
class ResetPool:
	states: jax.Array
	energies: jax.Array
	best_states: jax.Array
	best_energies: jax.Array
	magnetizations: jax.Array

@struct.dataclass
class EnvState:
	time: int
	s: jax.Array
	e: jax.Array
	best_s: jax.Array
	best_e: jax.Array
	temp: jax.Array

from modules.nmc_types import *

@struct.dataclass
class EnvParams:	
	max_steps_in_episode: int = 10000		#maximum number of iteration before resetting the env 

	nmc_Nsw_init: int = struct.field(default = 1000, pytree_node=False)

	nmc_Nsw_nbb: int = struct.field(default = 50, pytree_node=False)
	nmc_Nsw_bb: int = struct.field(default = 100, pytree_node=False)
	nmc_Nsw_eq: int = struct.field(default = 100, pytree_node=False)

	nmc_xT: float = 5.0
	Ti: float = 1.0
	T: float = 0.5
	Tf: float = 0.1
	
	nmc_schedule: bool = struct.field(default = True, pytree_node=False)

	nmc_init_annealing: bool = struct.field(default = True, pytree_node=False)
	nmc_eq_annealing: bool = struct.field(default = False, pytree_node=False)

	nmc_rand_jump: bool = struct.field(default = False, pytree_node=False)
	nmc_Ncycles: int = struct.field(default = 10, pytree_node=False)

	# lbp beta is currently fixed
	lbp_beta: float = struct.field(default = 1.0, pytree_node=False) 
	# lbp number message passing iterations
	lbp_num_iters: int = struct.field(default = 100, pytree_node=False)		
	# lbp tolerance for message diff
	lbp_tolerance_m: float = 1e-3		
	# lbp tolerance for distance from pinning
	lbp_tolerance_d: float = 0.01		
	# lbp convexify control parameters
	lbp_lambdas: jax.Array = struct.field(default_factory = lambda: jnp.array([0.1, 0.01, 0.5]))
      
	
class NMCgym(environment.Environment):
	"""
	JAX compatible NMC environment used for both NSA and RLNSA algorithms
	Has a specific implementation of the restarts to effiently support JAX vectorization:
	the restart states are pulled from a pool of precomputed states.
	"""

	def __init__(self,
							 Jh: Union[Jhdata, cnfdata],
							 energy_scaler: float,
							 lbp_beta: float,
							 nmc_schedule: bool,
							 track_stats: bool,
							 bench_mode: bool,
							 approximation: float,
							 pmode_idx: int, #0 for pubo, 1 for pising, 2 for cnf, 3 for wcnf
							 ground_state: Union[jax.Array, None], 
							 Jh_cat = None):

		super().__init__()
	
		self.Jh = Jh

		#K = number of independent subgraphs of the problem given
		if self.Jh.h.ndim > 1:
			self.K = self.Jh.h.shape[0]
			if Jh_cat is None:
				raise RuntimeError("Please provide Jh_cat, since K > 1")
			else:
				Jh_lbp = Jh_cat
		else:
			self.K = 1
			Jh_lbp = self.Jh

		self.pmode_idx = pmode_idx

		self.track_stats = track_stats

		self.gs = ground_state
		self.approx = approximation

		self.bench_mode = bench_mode #do not terminate and reset if GS is found (not for training mode)

		self.N = Jh_lbp.h.size

		self.energy_scaler = energy_scaler
		self.lbp_beta = lbp_beta

		#factor graph and other stuff for loopy belief propagation
		self.FG = lbp.init_factor_graph(Jh_lbp.J, 
																	  Jh_lbp.h, 
																		self.lbp_beta, 
																		self.pmode_idx)
		
		self.BP = infer.build_inferer(self.FG.bp_state, backend="bp")

		self.nmc_schedule = nmc_schedule

		#setup the epsilon scale of the interactions
		if isinstance(Jh_lbp, Jhdata):
			self.epsilon = jnp.sum(jnp.concatenate([jnp.sum(jnp.abs(j.data), axis=1, keepdims = True) 
																					 for j in Jh_lbp.Jat], axis=1), axis = 1)
		else:
			self.epsilon = jnp.sum(jnp.concatenate([jnp.sum(jnp.where(j[:, :, 0] != 0, 1, 0), axis = 1) 
																					 for j in Jh_lbp.Jat], axis = 1), axis = 1)

		self.epsilon += jnp.abs(Jh_lbp.h)
	
	@partial(jax.jit, static_argnums=(0,))
	def step(self,
					 key: jax.Array,
					 state: EnvState,
					 action: jax.Array,
					 reset_pool: ResetPool,
					 params: Optional[EnvParams]) -> Tuple[EnvObs, EnvState, jax.Array, jax.Array, Dict[Any, Any]]:
		"""
		Performs the NMC step of the environment (higher level); 
		if termination encountered, reset from the reset_pool of states
		"""

		key, key_reset = jax.random.split(key)
		obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)

		if not self.bench_mode:
			obs_re, state_re = self.reset_env(key_reset, reset_pool, params)

			# Auto-reset environment based on termination
			state = jax.tree.map(
					lambda x, y: jax.lax.select(done, x, y), state_re, state_st
			)
			obs = jax.tree.map(
					lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
			)
		else:
			obs = obs_st
			state = state_st

		return obs, state, reward, done, info

	#jit compiled by step of the parent class
	def step_env(self,
							 key,
							 state: EnvState,
							 action: jax.Array,
							 params: EnvParams) -> Tuple[EnvObs, EnvState, jax.Array, jax.Array, Dict[Any, Any]]:
		"""
		Performs the NMC step of the environment (lower level); 
		implements the logic of the NMC cycles and the magnetization/loopy belief propagation calculations
		"""
		policy_stats = {
			"bb_size": action.sum()/action.size
		}

		xT = jnp.where(action > 0, params.nmc_xT, 1.).astype(jnp.float32)

		old_best_e = state.best_e

		if self.nmc_schedule:
			beta = 1.0/params.T + (1.0/params.Tf - 1.0/params.T)*state.time/(params.max_steps_in_episode - 1 + 1e-9)
		else:
			beta = 1.0/params.T

		def run_nmc_cycles(Jh, s, e, best_s, best_e, key, xT):
			minskey = Minskey(s, e, key)
			result = nmc.nmc_cycles(Jh,  
															minskey, 
															
															best_s, 
															best_e, 

															xT, 
															1.0/beta,
															params.Tf,

															params.nmc_Ncycles, 
															params.nmc_Nsw_bb,
															params.nmc_Nsw_nbb,
															params.nmc_Nsw_eq,

															eq_annealing = params.nmc_eq_annealing, 
															rnd_jump = params.nmc_rand_jump, 
															skip_frozen = False,
															pmode_idx = self.pmode_idx,
															totrack = self.track_stats)

			s = result[0].min_s
			e = result[0].min_e
			best_s = result[1]
			best_e = result[2]
			mc_stats = result[3]

			return s, e, best_s, best_e, mc_stats

		if self.K == 1:
			s, e, best_s, best_e, mc_stats = run_nmc_cycles(self.Jh, 
																									 		state.s, 
																											state.e.reshape(()), 
																											state.best_s, 
																											state.best_e.reshape(()), 
																											key, xT)
			e = e.reshape((1,))
			best_e = best_e.reshape((1,))

		else:
			keys = jax.random.split(key, self.K)
			s, e, best_s, best_e, mc_stats = \
				jax.vmap(run_nmc_cycles)(self.Jh, 
														 		 state.s.reshape(self.K, -1), 
																 state.e, 
																 state.best_s.reshape(self.K, -1), 
																 state.best_e, 
																 keys, 
																 xT.reshape(self.K, -1))
			s = s.reshape(self.N)
			best_s = best_s.reshape(self.N)
			mc_stats = jax.tree.map(lambda x: x.mean(axis = 0), mc_stats)

		if NO_LBP:
			if self.K == 1:
				de_vector = jax.vmap(nmc.get_de, in_axes = (0, 0, 0, None, None))(self.Jh.Jat, self.Jh.h, s, s, self.pmode_idx)

			else:
				de_vector = jax.vmap(jax.vmap(nmc.get_de, in_axes = (0, 0, 0, None, None)), 
														 in_axes = (0, 0, 0, 0, None))(self.Jh.Jat, 
														 											self.Jh.h, 
																									s.reshape(self.K, -1), 
																									s.reshape(self.K, -1), self.pmode_idx)
														 
				de_vector = de_vector.reshape(-1)

			magnetizations = jnp.where(de_vector <= 0, 
																 jnp.array(0, dtype = de_vector.dtype), 
																 de_vector/2)
			lbp_stats = {}
			grad_stats = {}
			grad_stats["min_de"] = jnp.min(de_vector)

		else:
			beliefs, lbp_stats = lbp.surrogate_LBP(self.BP, 
																						self.FG.variable_groups[0],
																						s,
																						self.epsilon, 
																						self.lbp_beta,
																						params.lbp_num_iters, 
																						params.lbp_tolerance_m,
																						params.lbp_tolerance_d, 
																						params.lbp_lambdas)

			#absolute magnetizations for the GNN policy
			magnetizations = jnp.abs(beliefs[self.FG.variable_groups[0]][:, 0] - beliefs[self.FG.variable_groups[0]][:, 1])/beta

			grad_stats = {}

		# Update state dict and evaluate termination conditions
		state = EnvState(
			time = state.time + 1,
			s = s,
			e = e,
			best_s = best_s, 
			best_e = best_e,
			temp = jnp.array(1.0/beta)
		)

		done_approx, done_steps = self.is_terminal(state, params)
		done = jnp.logical_or(done_approx, done_steps)

		# intermediate improvement energy reward
		reward = jnp.where(old_best_e - best_e > 0, (old_best_e - best_e).astype(jnp.float32), 
																								 jnp.full_like(best_e, 0, dtype=jnp.float32)).sum()

		return (
				lax.stop_gradient(self.get_obs(state, magnetizations)),
				lax.stop_gradient(state),
				reward,

				done,
				{
					"policy_stats": policy_stats, 
					"grad_stats": grad_stats, 
					"lbp_stats": lbp_stats, 
					"mc_stats": mc_stats
				},
		)

	@partial(jax.jit, static_argnums=(0,))
	def reset(self, 
					  key: jax.Array, 
						reset_pool: ResetPool,
						params: EnvParams) -> Tuple[EnvObs, EnvState]:
		"""
		Performs the NMC reset of the environment (higher level)
		"""				

		obs, state = self.reset_env(key, reset_pool, params)

		return obs, state

	#jit compiled by reset of the parent class
	def reset_env(self, 
								key: jax.Array, 
								reset_pool: ResetPool,
								params: EnvParams) -> Tuple[EnvObs, EnvState]:
		"""
		Performs the NMC reset of the environment (lower level);
		the reset is taking the state from the precomputed pool of states
		"""		
		i = jax.random.choice(key, a = reset_pool.energies.size)
		
		magnetizations = reset_pool.magnetizations[i]

		state = EnvState(
			time = 0,
			s = reset_pool.states[i],
			e = reset_pool.energies[i],
			best_s = reset_pool.best_states[i], 
			best_e = reset_pool.best_energies[i],
			temp = jnp.array(params.T, dtype = jnp.float32)
		)

		return(
			lax.stop_gradient(self.get_obs(state, magnetizations)), 
			lax.stop_gradient(state)
		)
	
	def reset_env_idx(self, 
										idx: jax.Array, 
										reset_pool: ResetPool,
										params: EnvParams) -> Tuple[EnvObs, EnvState]:
		
		magnetizations = reset_pool.magnetizations[idx]

		state = EnvState(
			time = 0,
			s = reset_pool.states[idx],
			e = reset_pool.energies[idx],
			best_s = reset_pool.best_states[idx], 
			best_e = reset_pool.best_energies[idx],
			temp = jnp.array(params.T)
		)

		return(
			lax.stop_gradient(self.get_obs(state, magnetizations)), 
			lax.stop_gradient(state)
		)

	def reset_pool(self, 
								 key: jax.Array, 
								 pool_size: int,
								 min_start: bool,
								 params: EnvParams, 
								 p_devices: int = 1) -> ResetPool:
		"""
		Precomputes the reset states for the reset_pool; 
		retuired for jax since the episode length can be arbitrary, and performing 
		simulated annealing reset at each step of the algorithm is very costly
		"""		
		key, subkey = jax.random.split(key)

		pool_s = jax.random.choice(subkey, jnp.array([False, True]), 
															 shape = (p_devices, pool_size//p_devices, self.N))

		def get_sme(init_s: jax.Array, key: jax.Array):
			def get_se(Jh, init_s, key):
				init_e = nmc.get_energy(Jh, init_s, self.pmode_idx)
				sminskey = Sminskey(init_s, init_s, init_e, init_e, key)

				if params.nmc_init_annealing:
					beta_sa = jnp.array([1.0/params.Ti, 1.0/params.T], dtype=jnp.float32)
					sminskey, nmc_stats = nmc.sa_annealing(Jh, 
																									sminskey, 
																									beta_sa, 
																									params.nmc_Nsw_init, 
																									self.pmode_idx, 
																									self.track_stats)
				else:
					Teq = jnp.full_like(init_s, params.T, dtype=jnp.float32)
					sminskey, nmc_stats = nmc.nmc_sampling(Jh, 
																								 sminskey, 
																								 Teq, 
																								 params.nmc_Nsw_init,
																								 skip_frozen = False, 
																								 pmode_idx = self.pmode_idx, 
																								 totrack = self.track_stats)
				if min_start:
					s = sminskey.min_s
					e = sminskey.min_e
				else:
					s = sminskey.s
					e = sminskey.e

				best_s = sminskey.min_s
				best_e = sminskey.min_e

				return s, e, best_s, best_e, nmc_stats				

			if self.K == 1:
				s, e, best_s, best_e, nmc_stats = get_se(self.Jh, init_s, key)
				e = e.reshape((1,))
				best_e = best_e.reshape((1,))

			else:
				keys = jax.random.split(key, self.K)
				s, e, best_s, best_e, nmc_stats = jax.vmap(get_se)(self.Jh, init_s.reshape(self.K, -1), keys)
				s = s.reshape(self.N)
				best_s = best_s.reshape(self.N)
				nmc_stats = jax.tree.map(lambda x: x.mean(axis = 0), nmc_stats)

			if NO_LBP: #for performance during testing on a cpu
				if self.K == 1:
					de_vector = jax.vmap(nmc.get_de, in_axes = (0, 0, 0, None, None))(self.Jh.Jat, 
																																			 			self.Jh.h, 
																																						s, 
																																						s, 
																																						self.pmode_idx)

				else:
					de_vector = jax.vmap(jax.vmap(nmc.get_de, in_axes = (0, 0, 0, None, None)), 
																in_axes = (0, 0, 0, 0, None))(self.Jh.Jat, 
																										 					self.Jh.h, 
																										 					s.reshape(self.K, -1), 
																										 					s.reshape(self.K, -1), 
																															self.pmode_idx)
					de_vector = de_vector.reshape(-1)

				magnetizations = jnp.where(de_vector <= 0, 
																	 jnp.array(0, dtype = de_vector.dtype), 
																	 de_vector/2)
			else:
				beliefs, _ = lbp.surrogate_LBP(self.BP, 
																			 self.FG.variable_groups[0],
																			 s,
																			 self.epsilon, 
																			 self.lbp_beta,
																			 params.lbp_num_iters, 
																			 params.lbp_tolerance_m,
																			 params.lbp_tolerance_d, 
																			 params.lbp_lambdas)

				#absolute magnetizations for the GNN policy
				magnetizations = jnp.abs(beliefs[self.FG.variable_groups[0]][:, 0] - beliefs[self.FG.variable_groups[0]][:, 1])*params.T

			sme = ResetPool(
				states = s,
				energies = e,
				best_states = best_s,
				best_energies = best_e,
				magnetizations = magnetizations #type: ignore
			)
			return sme, nmc_stats

		def p_get_sme(init_s: jax.Array, keys: jax.Array):
			keys = jax.random.split(key, pool_size//p_devices)
			return jax.vmap(get_sme)(init_s, keys)

		keys = jax.random.split(key, p_devices)
		if p_devices > 1:
			reset_pool, nmc_stats = jax.pmap(p_get_sme)(pool_s, keys)
		else:
			reset_pool, nmc_stats = jax.jit(jax.vmap(p_get_sme))(pool_s, keys)

		return reset_pool

	@partial(jax.jit, static_argnums = (0))
	def get_obs(self, 
						 	state: EnvState, 
							magnetizations: jax.Array) -> EnvObs:
		"""
		Computes observations that can potentially be passed to the recurrent policy
		"""
		N0 = self.N//self.K

		return EnvObs(
			s = state.s.astype(jnp.float32),
			r = magnetizations.astype(jnp.float32),

			e = (state.e - state.best_e).astype(jnp.float32)/self.energy_scaler,
			best_e = (state.best_e).astype(jnp.float32)/self.energy_scaler,

			dtobest = (state.s != state.best_s).reshape(self.K, N0).sum(axis = 1)/N0,

			temp = state.temp.reshape(1).astype(jnp.float32)
		)


	def is_terminal(self, state: EnvState, params: EnvParams) -> Any:
		"""
		Checks if the episode is terminated [approximation met or the episode is ended (finite horizon)
		"""
		# Check termination criteria: ground state found if known
		if self.gs is None:
			done1 = False
		else:
			done1 = jnp.all((state.best_e - self.gs - self.approx) < jnp.finfo(jnp.float16).eps)

		# Check number of steps in episode termination condition
		done_steps = state.time >= params.max_steps_in_episode

		return done1, done_steps


	@property
	def name(self) -> str:
		return "RLNMC-v1"

	def action_space(self, params: Optional[EnvParams] = None):
		#to implement these if necessary
		raise NotImplementedError

	def observation_space(self, params: EnvParams):
		#to implement these if necessary
		raise NotImplementedError

	def state_space(self, params: EnvParams):
		#to implement these if necessary
		raise NotImplementedError

class GymnaxWrapper(object):
  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

@struct.dataclass
class LogState:
  env_state: EnvState

  episode_returns: float
  episode_lengths: int

  returned_episode_returns: float
  returned_episode_lengths: int

  timestep: int


class LogWrapper(GymnaxWrapper):

		def __init__(self, env: environment.Environment):
				super().__init__(env)

		@partial(jax.jit, static_argnums=(0,))
		def reset(self, 
							key: jax.Array, 
							reset_pool: ResetPool,
							params: Optional[environment.EnvParams] = None) -> Tuple[EnvObs, LogState]:

			obs, env_state = self._env.reset(key, reset_pool, params)
			
			state = LogState(env_state, 0, 0, 0, 0, 0)

			return obs, state

		def reset_env_idx(self, 
											idx: jax.Array, 
											reset_pool: ResetPool, 
											params: EnvParams) -> Tuple[EnvObs, LogState]:
			
			obs, env_state = self._env.reset_env_idx(idx, reset_pool, params)
			
			state = LogState(env_state, 0, 0, 0, 0, 0)

			return obs, state

		@partial(jax.jit, static_argnums=(0,))
		def step(self,
						 key: jax.Array,
						 state: LogState,
						 action: jax.Array,
						 reset_pool: ResetPool,
						 params: Optional[EnvParams] = None) -> Tuple[EnvObs, LogState, jax.Array, jax.Array, Dict[Any, Any]]:

			# for tracking the jump
			old_e = state.env_state.e
			old_s = state.env_state.s
			old_best_e = state.env_state.best_e
			old_best_s = state.env_state.best_s

			# make the environment step
			obs, env_state, reward, done, info = self._env.step(
					key, state.env_state, action, reset_pool, params
			)

			new_episode_return = state.episode_returns + reward
			new_episode_length = state.episode_lengths + 1

			state = LogState(
					env_state = env_state,

					episode_returns = new_episode_return*(1 - done),
					episode_lengths = new_episode_length*(1 - done),
					
					returned_episode_returns = state.returned_episode_returns*(1 - done)
						+ new_episode_return*done,
					returned_episode_lengths = state.returned_episode_lengths*(1 - done)
						+ new_episode_length*done,

					timestep = state.timestep + 1
			)

			info["episode_returns"] = state.episode_returns
			info["episode_lengths"] = state.episode_lengths

			info["returned_episode_returns"] = state.returned_episode_returns
			info["returned_episode_lengths"] = state.returned_episode_lengths
			info["returned_episode"] = done

			info["timestep"] = state.timestep

			# logging the trajectory of energies and observations
			info["trajectory"] = {}
			info["trajectory"]["de"] = state.env_state.e - old_e
			info["trajectory"]["d_best_e"] = state.env_state.best_e - old_best_e

			if self._env.gs is not None:
				info["trajectory"]["current_e"] = state.env_state.e - self._env.gs
				info["trajectory"]["best_e"] = state.env_state.best_e - self._env.gs
			else:
				info["trajectory"]["current_e"] = state.env_state.e
				info["trajectory"]["best_e"] = state.env_state.best_e

			#the jump between the current minima
			info["trajectory"]["jump"] = (old_s != state.env_state.s).sum()/state.env_state.s.size
			#the jump between the best minima
			info["trajectory"]["best_jump"] = (
				(old_best_s != state.env_state.best_s).sum()/state.env_state.best_s.size
			) 
			# current distance to the best state
			info["trajectory"]["dtobest"] = obs.dtobest
			info["trajectory"]["temp"] = obs.temp

			info["trajectory"]["bb_size"] = action.sum()/action.size
			info["trajectory"]["bb_r_mean"] = (
				jnp.where(action == 1, obs.r, 0.0).sum()/(action.sum() + 1e-9)
			)

			info["trajectory"]["avg_r"] = jnp.mean(obs.r)
			info["trajectory"]["max_r"] = jnp.max(obs.r)
			info["trajectory"]["min_r"] = jnp.min(obs.r)

			if bool(info["lbp_stats"]):
				info["trajectory"]["lbp_init_distance"]  = info["lbp_stats"]["init_distance"]
				info["trajectory"]["lbp_final_distance"] = info["lbp_stats"]["final_distance"]
				info["trajectory"]["lbp_init_steps"] 		 = info["lbp_stats"]["init_steps"]
				info["trajectory"]["lbp_final_steps"] 	 = info["lbp_stats"]["final_steps"]

			if bool(info["grad_stats"]):
				info["trajectory"]["min_de"] = info["grad_stats"]["min_de"]

			return obs, state, reward, done, info
