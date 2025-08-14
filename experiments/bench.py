import os
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=8" #uncomment on macos for multi-cpu processing

import sys
import time
import json
import argparse

sys.path.append('.')

from modules.nmc_types import Jhdata

from modules.policies.nmc_policy import *
from modules.policies.nmc_base import *
from modules.lbp import *
from experiments.callbacks import *
from modules.environment.nmcgym import *
from modules.utils import tts_bootstrap
from modules.diversity import get_diversity

import jax
import jax.numpy as jnp
from typing import NamedTuple, Any, Dict, Union

from flax import nnx
import distrax

import pickle
import orbax.checkpoint as ocp
from problems.readers import *

def bench_sa(Jh_list: List[Jhdata], # stacked instances
						 gs_list: Union[List[jax.Array], None],
						 approximation: float,
						 names: List[str], prefix: str,
						 key: jax.Array,
						 track_stats: bool,
						 pmode_idx: int, 
						 config: Dict,
						 devices, 
						 succ_folder_name) -> Tuple[Sminskey, Any, Any, Any]:
	"""
	Benchmarking function of the Simulated Annealing algorithm
	Many instances (of the same size) and many replicas can be processed vectorized in parallel 
	across multiple (CPU or GPU) devices
	"""

	save_dir = os.path.abspath(os.path.join(config["LOG"]["PATH"], 
																					config["LOG"]["PROJECT_NAME"]))
	os.makedirs(save_dir, exist_ok = True)

	if len(config["DIVERSITY_R"]) > 0:
		track_diversity = True
		diversity_per_replica = config["DIVERSITY_PER_REPLICA"]
	else:
		track_diversity = False
		diversity_per_replica = False

	if config["SAVE_RESULTS"]:
		if succ_folder_name is None:
			suc_dir = os.path.join(save_dir, "suc")
		else:
			suc_dir = os.path.join(save_dir, succ_folder_name)
			
		#make the folder where to save the success values for each instance
		os.makedirs(suc_dir, exist_ok = True)

		for name in names:
			with open(os.path.join(suc_dir, name + "_suc.dat"), mode = 'w') as f:
				pass
			with open(os.path.join(suc_dir, name + "_beste.dat"), mode = 'w') as f:
				pass

			if track_diversity:
				with open(os.path.join(suc_dir, name + "_div.dat"), mode = 'w') as f:
					pass
	
	timer = 0
	
	NDEVICES = len(devices)

	N = Jh_list[0].h.size
	K_ALL = len(Jh_list)
	
	assert K_ALL%NDEVICES == 0, "Can't distribute equally across devices!"
	K_pd = K_ALL//NDEVICES

	Jh_stacked = pad_and_stack_Jh(Jh_list)

	if gs_list is not None:
		ground_states = jnp.stack(gs_list)
	else:
		ground_states = None

	Jh_stacked_pd = \
		jax.tree.map(lambda x: jnp.stack([x[i:i + K_pd] for i in range(0, K_ALL, K_pd)], 0), Jh_stacked)
	
	NUM_REPLICAS = config["NUM_REPLICAS"]

	key, subkey = jax.random.split(key) #jax random keys
	s = jax.random.choice(subkey, jnp.array([False, True]), shape=(NDEVICES, K_pd, NUM_REPLICAS, N))

	def get_sminskey(Jh, init_s):
		if pmode_idx == 0:
			init_e = -nmc.Jxs_pubo(Jh.J, Jh.h, init_s)
		elif pmode_idx == 1:
			init_e = -nmc.Jxs_pising(Jh.J, Jh.h, init_s)
		elif pmode_idx == 2:
			init_e = -nmc.Jxs_cnf(Jh.J, Jh.h, init_s)
		else:
			init_e = -nmc.Jxs_wcnf(Jh.J, Jh.h, init_s)

		sminskey = Sminskey(init_s, init_s, init_e, init_e, key)
		return sminskey
	sminskey = jax.vmap(jax.vmap(jax.vmap(get_sminskey, in_axes = (None, 0))))(Jh_stacked_pd, s)
	
	best_e_track = sminskey.min_e
	best_s_track = sminskey.min_s
	
	best_e_reset = sminskey.min_e
	best_s_reset = sminskey.min_s

	def sa_schedule(count):
		frac = count/config["SA"]["TOTAL_SWEEPS"]
		return frac
	
	assert config["NUM_LOG_STEPS"] > 0
	
	beta_init  = 1.0/config["SA"]["T_init"]
	beta_final = 1.0/config["SA"]["T_final"]
	betaf = beta_init

	if track_diversity:
		set_of_all_solutions = [set() for _ in range(K_ALL)]
		if diversity_per_replica:
			set_of_solutions_pr  = []
			for _ in range(K_ALL):
				set_of_solutions_pr.append([set() for _ in range(NUM_REPLICAS)])

	NUM_SWEEPS_PER_LOG = config["SA"]["TOTAL_SWEEPS"]//config["NUM_LOG_STEPS"]
	
	all_log_info = []
	all_tts_results = []
	all_other = []
	for log_step in range(config["NUM_LOG_STEPS"]):
		#The SA schedule is linearly decreasing
		betai = betaf
		betaf = beta_init + sa_schedule((log_step + 1)*NUM_SWEEPS_PER_LOG)*(beta_final - beta_init)

		sminskey_old = sminskey
		sminskey = Sminskey(sminskey.s, best_s_reset, sminskey.e, best_e_reset, sminskey.key)

		timer_old = timer
		time_a = time.time()
		if NDEVICES == 1:
			sminskey, mc_stats = \
					jax.vmap(jax.vmap(jax.vmap(nmc.sa_annealing,
						in_axes = (None, 0, None, None, None, None)),
						in_axes = (0, 0, None, None, None, None)),
						in_axes = (0, 0, None, None, None, None))(Jh_stacked_pd, 
																											sminskey, 
																											jnp.array([betai, betaf]),
																											NUM_SWEEPS_PER_LOG, 
																											pmode_idx, 
																											track_stats)		
		else:
			sminskey, mc_stats = \
					jax.pmap(jax.vmap(jax.vmap(nmc.sa_annealing,
						in_axes = (None, 0, None, None, None, None)),
						in_axes = (0, 0, None, None, None, None)),
						in_axes = (0, 0, None, None, None, None), 
						static_broadcasted_argnums = (3, 4, 5))(Jh_stacked_pd, 
																										sminskey, 
																										jnp.array([betai, betaf]),
																										NUM_SWEEPS_PER_LOG, 
																										pmode_idx, 
																										track_stats)		
		jax.tree.map(lambda x: x.block_until_ready(), sminskey)
		timer += time.time() - time_a

		def get_jump_stats(skey_old: Sminskey, skey: Sminskey):
			stats = {}
			stats["de"] = skey.e - skey_old.e; stats["jump"] = (skey_old.s != skey.s).sum()/skey.s.size
			stats["min_de"] = skey.min_e - skey_old.min_e; stats["min_jump"] = (skey_old.min_s != skey.min_s).sum()/skey.min_s.size
			return stats
		jump_stats = jax.vmap(jax.vmap(jax.vmap(get_jump_stats)))(sminskey_old, sminskey)

		mc_stats.update(jump_stats) 

		def update_best(e: jax.Array, s: jax.Array, skey: Sminskey):
			min_e, min_s = lax.cond(skey.min_e < e,
													    lambda: (skey.min_e, skey.min_s), 
													    lambda: (e, s))
			return min_e, min_s
		
		best_e_track, best_s_track = jax.vmap(jax.vmap(jax.vmap(update_best)))(best_e_track, 
																																				 	 best_s_track, 
																																					 sminskey)

		if gs_list is not None:
			energies = sminskey.e - jnp.reshape(ground_states, (NDEVICES, K_pd, 1))
			best_energies = best_e_track - jnp.reshape(ground_states, (NDEVICES, K_pd, 1))
		else:
			energies = sminskey.e
			best_energies = best_e_track

		if track_diversity:
			time_diversity = 0
			time_div_a = time.time()
			best_energies_np = np.array(best_energies).reshape(K_ALL, NUM_REPLICAS)
			best_states_np = np.array(best_s_track.reshape(K_ALL, NUM_REPLICAS, N))

			for i in range(K_ALL):
				for j in range(NUM_REPLICAS):
					if best_energies_np[i][j] <= approximation + jnp.finfo(jnp.float16).eps:
						#solutions for diversity across all replicas
						set_of_all_solutions[i].add(tuple(best_states_np[i][j]))

						if diversity_per_replica:
							#solutions for diversity in each replica
							set_of_solutions_pr[i][j].add(tuple(best_states_np[i][j]))

			time_diversity += time.time() - time_div_a
	
		log_info = {"coarse": {"trajectory": {}}}
		log_info["coarse"]["mc_stats"] = mc_stats
		log_info["coarse"]["trajectory"]["current_e"] = energies
		log_info["coarse"]["trajectory"]["best_e"] = best_energies
		log_info["coarse"]["trajectory"]["dtobest"] = ((sminskey.min_s != best_s_track).sum(axis = 3)/N)

		metric = {}
		
		if track_diversity:
			diversity = {r: {} for r in config["DIVERSITY_R"]}
			diversity["int"] = {}

			time_div_a = time.time()
			for name, solutions in zip(names, set_of_all_solutions):
				div_int, div_r = get_diversity(solutions, config["DIVERSITY_R"])
				
				#per r diversity
				for ri, r in enumerate(config["DIVERSITY_R"]):
					diversity[r][name] = div_r[ri]

				diversity["int"][name] = div_int

			for r in config["DIVERSITY_R"]:
				diversity_values = np.array(list(diversity[r].values()))
				
				metric[f"div_median/{r}"] = np.median(diversity_values)
				metric[f"div_mean/{r}"] = np.mean(diversity_values)
				metric[f"div_min/{r}"] = np.min(diversity_values)
				metric[f"div_max/{r}"] = np.max(diversity_values)
			
			metric[f"div_median/int"] = np.median(np.array(list(diversity["int"].values())))
			metric[f"div_mean/int"] = np.mean(np.array(list(diversity["int"].values())))
			metric[f"div_min/int"] = np.min(np.array(list(diversity["int"].values())))
			metric[f"div_max/int"] = np.max(np.array(list(diversity["int"].values())))

			time_diversity += time.time() - time_div_a

			num_solutions = {}
			for name, solutions in zip(names, set_of_all_solutions):
				num_solutions[name] = len(solutions)
			num_sol_values = np.array(list(num_solutions.values()))

			metric["num_sol_median"] = np.median(num_sol_values)
			metric["num_sol_mean"] = np.mean(num_sol_values)
			metric["num_sol_min"] = np.min(num_sol_values)
			metric["num_sol_max"] = np.max(num_sol_values)

		if diversity_per_replica:
			time_div_a = time.time()
			div_array = np.empty((NDEVICES, K_pd, NUM_REPLICAS))
			num_sol_array = np.empty((NDEVICES, K_pd, NUM_REPLICAS))
			for i in range(K_ALL):
				for j in range(NUM_REPLICAS):
					div_int, _ = get_diversity(set_of_solutions_pr[i][j], config["DIVERSITY_R"])
					div_array[i//K_pd, i%K_pd, j] = div_int
					num_sol_array[i//K_pd, i%K_pd, j] = len(set_of_solutions_pr[i][j])
			
			time_diversity += time.time() - time_div_a
			log_info["coarse"]["trajectory"]["div"] = div_array
			log_info["coarse"]["trajectory"]["num_sol"] = num_sol_array

		all_log_info.append(log_info)

		other = {}
		other["betai"] = betai
		other["betaf"] = betaf

		other["total_seconds_elapsed"] = jnp.array(timer)
		other["seconds_per_log"] = jnp.array(timer - timer_old)

		if track_diversity:
			other["seconds_per_diversity"] = time_diversity

		LOG_SWEEP = (log_step + 1)*NUM_SWEEPS_PER_LOG

		other["total_sweeps_elapsed"] = jnp.array(LOG_SWEEP)
		other["sweeps_per_second"] = jnp.array(NUM_SWEEPS_PER_LOG)/other["seconds_per_log"]

		all_other.append(other)

		tts_results = tts_bootstrap(best_energies.reshape((K_ALL, -1)), 
																jnp.full((K_ALL,), LOG_SWEEP),
																jnp.full((K_ALL,), approximation),
																jax.random.key(0),
																jnp.array(config["PERCENTILES"]), 
																config["BOOTSTRAP"])
		all_tts_results.append(tts_results)

		successes = {}
		bestes = {}

		if config["LOG"]["WANDB"]:
			if K_ALL > 1:
				metric["pos"] = wandb.Histogram(tts_results["successes"]/(tts_results["successes"] + tts_results["failures"]))
				metric["tts_means"] = wandb.Histogram(tts_results["tts_means"])
			else:
				metric["pos"] = tts_results["successes"]/(tts_results["successes"] + tts_results["failures"])
				metric["tts_means"] = tts_results["tts_means"]

		else:
			if K_ALL > 1:
				metric["pos"] = jnp.histogram(tts_results["successes"]/(tts_results["successes"] + tts_results["failures"]))
				metric["tts_means"] = jnp.histogram(tts_results["tts_means"])
			else:
				metric["pos"] = tts_results["successes"]/(tts_results["successes"] + tts_results["failures"])
				metric["tts_means"] = tts_results["tts_means"]
		
		for name, suc_value, beste in zip(names, 
																			tts_results["successes"], 
																			best_energies.reshape((K_ALL, -1))):
			successes[name] = suc_value
			bestes[name] = beste

		if config["SAVE_RESULTS"]:
			for name in names:
				with open(os.path.join(suc_dir, name + "_suc.dat"), mode = 'a') as f:
					np.savetxt(f, np.array([LOG_SWEEP, successes[name]], dtype = int), 
										 newline = " ", 
										 fmt = "%i")

				with open(os.path.join(suc_dir, name + "_beste.dat"), mode = 'a') as f:
					np.savetxt(f, np.concatenate([[LOG_SWEEP], bestes[name].astype(int)], dtype = int), 
										 newline = " ", 
										 fmt = "%i")

				if track_diversity:
					with open(os.path.join(suc_dir, name + "_div.dat"), mode = 'a') as f:
						np.savetxt(f, np.concatenate([[LOG_SWEEP], 
																					[diversity[r][name] for r in config["DIVERSITY_R"]], 
																					[diversity["int"][name]]], dtype = float), 
											newline = " ", 
											fmt = "%.7g")

		for i, p in enumerate(config["PERCENTILES"]):
			metric[f"tts_{p}"] = tts_results["tts_p_means"][i]
			metric[f"s_tts_{p}"] = tts_results["tts_p_stds"][i]

		#here to log the stats
		if config["LOG"]["WANDB"]:
			for i, name in enumerate(names):
				bench_wandb_callback(LOG_SWEEP, None,
														 jax.tree.map(lambda x: x.reshape(-1, x.shape[2])[i].reshape(1, -1), log_info), 
														 metric if i == 0 else None,
														 other if i == 0 else None,
														 [name], prefix)

		else:
			pass
		
		#endfor log step
		
	bench_results = sminskey, all_log_info, all_tts_results, all_other

	return bench_results


def make_bench_nmc(Jh_list: List[Jhdata], 
									 gs_list: Union[List[jax.Array], None],
									 energy_scaler: float,
									 approximation: float,
									 names: List[str], prefix: str,
									 track_stats: bool,
									 pmode_idx: int, 
									 config: Dict,
									 env_params: EnvParams,
									 devices,
									 ac_config, 
									 succ_folder_name):
	"""
	Benchmarking function of the (RL) Nonlocal Monte Carlo algorithm
	Many instances (of the same size) and many replicas can be processed vectorized in parallel 
	across multiple (CPU or GPU) devices (however, vectorization is not identical to the SA bench function)
	"""

	save_dir = os.path.abspath(os.path.join(config["LOG"]["PATH"], 
																					config["LOG"]["PROJECT_NAME"]))
	os.makedirs(save_dir, exist_ok = True)

	if len(config["DIVERSITY_R"]) > 0:
		track_diversity = True
		diversity_per_replica = config["DIVERSITY_PER_REPLICA"]
	else:
		track_diversity = False
		diversity_per_replica = False

	if config["SAVE_RESULTS"]:
		if succ_folder_name is None:
			suc_dir = os.path.join(save_dir, "suc")
		else:
			suc_dir = os.path.join(save_dir, succ_folder_name)
			
		#make the folder where to save the success values for each instance
		os.makedirs(suc_dir, exist_ok = True)

		for name in names:
			with open(os.path.join(suc_dir, name + "_suc.dat"), mode = 'w') as f:
				pass
			with open(os.path.join(suc_dir, name + "_beste.dat"), mode = 'w') as f:
				pass

			if track_diversity:
				with open(os.path.join(suc_dir, name + "_div.dat"), mode = 'w') as f:
					pass

	NDEVICES = len(devices)
	NUM_REPLICAS = config["NUM_REPLICAS"]

	K_ALL = len(Jh_list)
			
	assert NUM_REPLICAS%NDEVICES == 0, "Can't distribute equally across devices!"
	NUM_REP_PD = NUM_REPLICAS//NDEVICES
	
	if K_ALL > 1:
		Jh_cat = pad_and_concatenate_Jh(Jh_list, pmode_idx)
		Jh_stacked = pad_and_stack_Jh(Jh_list) 
	else:
		Jh_cat = Jh_list[0]
		Jh_stacked = Jh_list[0]

	N = Jh_cat.h.size

	if gs_list is not None:
		ground_states = jnp.stack(gs_list)
	else:
		ground_states = None

	#init the environment
	env = NMCgym(Jh = Jh_stacked, 
							 nmc_schedule = env_params.nmc_schedule,
							 energy_scaler = energy_scaler,
							 lbp_beta = env_params.lbp_beta, 
							 track_stats = track_stats,
							 bench_mode = True,
							 approximation = approximation,
							 pmode_idx = pmode_idx, 
							 ground_state = None, 
							 Jh_cat = Jh_cat)

	if config["ALGO"] == "rlnmc":
		model = FactorActorCritic(Jh_cat, pmode_idx, **ac_config, rngs = nnx.Rngs(0))
		model_graph, model_state_rnd, model_other = nnx.split(model, nnx.Param, ...)

		rngs = nnx.Rngs(0) #nnx random keys
		h_init = model.gru_cell.initialize_carry((NDEVICES, NUM_REP_PD, N, -1), rngs)
		if model.global_gru:
			h_global_init = model.global_gru_cell.initialize_carry((NDEVICES, NUM_REP_PD, K_ALL, -1), rngs)
		else:
			h_global_init = None


	NUM_REPLICAS = config["NUM_REPLICAS"]

	INIT_MC_SWEEPS = env_params.nmc_Nsw_init
	if env_params.nmc_rand_jump:
		STEP_MC_SWEEPS = (
      (1 + 1 + env_params.nmc_Nsw_eq)*env_params.nmc_Ncycles 
    )
		
	else:
		STEP_MC_SWEEPS = (
			((env_params.nmc_Nsw_bb + env_params.nmc_Nsw_nbb)//2 + 
				env_params.nmc_Nsw_eq)*env_params.nmc_Ncycles  
		)

	if track_diversity:
		set_of_all_solutions = [set() for _ in range(K_ALL)]
		if diversity_per_replica:
			set_of_solutions_pr  = []
			for _ in range(K_ALL):
				set_of_solutions_pr.append([set() for _ in range(NUM_REPLICAS)])

	def bench_nmc(key, 
								model_state_pre = None) -> Tuple[Any, Any, Any, Any]:
		"""
		Benchmarking function of NMC type algorithms; 
		split per device over the replicas (in SA function the split is over the instances)
		"""
		timer = 0

		#reset each replica environment
		key, reset_pool_key = jax.random.split(key)

		time_a = time.time()
		reset_pool = env.reset_pool(reset_pool_key, 
																NUM_REPLICAS,
																min_start = False, #to reset in the final state of the annealing instead of the best one
																params = env_params, 
																p_devices = NDEVICES) #first dimension is number of devices

		obsv, env_state = jax.vmap(jax.vmap(env.reset_env_idx, in_axes = (0, None, None)), 
																				in_axes = (0, 0, None))(jnp.stack([jnp.arange(NUM_REP_PD)]*NDEVICES), 
																													reset_pool, env_params)

		jax.tree.map(lambda x: x.block_until_ready(), (obsv, env_state))
		timer += time.time() - time_a
		
		if config["ALGO"] == "rlnmc":
			if model_state_pre is not None:
				#use the pre-trained model provided
				model_state = model_state_pre
			else:
				model_state = model_state_rnd

		elif config["ALGO"] == "nmc":
			def nmc_schedule(count):
				frac = count/config["NMC"]["TOTAL_STEPS_PER_REPLICA"]
				return frac

		else: 
			raise RuntimeError("Wrong ALGO!")
		
		
		if config["ALGO"] == "rlnmc":

			def rlnmc_per_device(env_on_device, env_params, runner_state, 
													 model_graph, model_state, model_other):
				def _rlnmc_step(runner_state, unused):
					env_state, last_obs, last_done, key, h, h_global = runner_state

					key, subkey = jax.random.split(key)

					# local observations
					x = jnp.stack((last_obs.s, last_obs.r), axis = 2)[None, :]


					extras = jnp.concatenate([jnp.tile(last_obs.temp[:, :,  None], (1, K_ALL, 1)), 
																						 last_obs.best_e[:, :, None]], axis = 2)[None, :] #shape [1, batch, K_ALL, 2]

					model = nnx.merge(model_graph, model_state, model_other)
					h, h_global, pi, _ = model(h, h_global, x, extras, 
																		#no resetting the gru states during benchmarking
																		jnp.zeros_like(last_done)[None, :])

					pi_bernoulli = distrax.Bernoulli(logits = pi.squeeze(axis = 0))
					action = pi_bernoulli.sample(seed = subkey)

					key, subkey = jax.random.split(key)
					step_keys = jax.random.split(subkey, NUM_REP_PD)

					#environment step
					obsv, env_state, _, done, info = \
						jax.vmap(env_on_device.step, in_axes = (0, 0, 0, None, None))(step_keys, env_state, action, None, env_params)

					info["energies"] = env_state.e
					info["best_energies"] = env_state.best_e
					info["dtobest"] = obsv.dtobest

					runner_state = (env_state, obsv, done, key, h, h_global)
					
					return runner_state, info
				
				runner_state, info = jax.lax.scan(_rlnmc_step, runner_state, None, NUM_STEPS_PER_LOG)

				return runner_state, info
			
			key, *subkeys = jax.random.split(key, NDEVICES + 1)
			runner_state = (
				env_state,
				obsv,
				jnp.zeros((NDEVICES, NUM_REP_PD), dtype=bool),
				jnp.stack(subkeys),
				h_init,
				h_global_init
			)

			#JAX pmap axes explicit
			_in_axes = (
				None, None,   # env, env_params
				(0, 0, 0, 0, 0, 0), # runner_state
				None, None, None)		# model_graph, model_state, model_other
		
		else:

			def nmc_per_device(env_on_device, env_params, runner_state):
				def _nmc_step(runner_state, unused):
					env_state, last_obs, last_done, key = runner_state

					key, subkey = jax.random.split(key)
					step_keys = jax.random.split(subkey, NUM_REP_PD)

					if config["NMC"]["RANDOM_BACKBONE"]:
						key, subkey = jax.random.split(key)
						action = random_backbones(last_obs.r, 
																			(1.0 - nmc_schedule(env_state.time[:, None]))*config["NMC"]["RND"], 
																			subkey)
					else:
						threshold = config["NMC"]["CUTOFF_start"] + \
																				nmc_schedule(env_state.time[:, None])*(config["NMC"]["CUTOFF_end"] - config["NMC"]["CUTOFF_start"])

						action = threshold_backbones(last_obs.r, threshold)

					#environment step
					obsv, env_state, _, done, info = \
						jax.vmap(env_on_device.step, in_axes = (0, 0, 0, None, None))(step_keys, env_state, action, None, env_params)

					info["energies"] = env_state.e
					info["best_energies"] = env_state.best_e
					info["dtobest"] = obsv.dtobest

					runner_state = (env_state, obsv, done, key)
					
					return runner_state, info
					
				runner_state, info = jax.lax.scan(_nmc_step, runner_state, None, NUM_STEPS_PER_LOG)

				return runner_state, info

			key, *subkeys = jax.random.split(key, NDEVICES + 1)
			runner_state = (
				env_state,
				obsv,
				jnp.zeros((NDEVICES, NUM_REP_PD), dtype = bool),
				jnp.stack(subkeys)
			)

			#JAX pmap axes explicit
			_in_axes = (None, None, 0)
			
		all_log_info = []
		all_tts_results = []
		all_other = []

		if config["NMC"]["TOTAL_STEPS_PER_REPLICA"] > config["NUM_LOG_STEPS"]:
			NUM_STEPS_PER_LOG = config["NMC"]["TOTAL_STEPS_PER_REPLICA"]//config["NUM_LOG_STEPS"]
		else:
			NUM_STEPS_PER_LOG = 1
			config["NUM_LOG_STEPS"] = config["NMC"]["TOTAL_STEPS_PER_REPLICA"]
		
		for log_step in range(config["NUM_LOG_STEPS"]):
			timer_old = timer
			time_a = time.time()
			if config["ALGO"] == "rlnmc":
				if NDEVICES == 1:
					a, b = jax.vmap(rlnmc_per_device, 
													in_axes = _in_axes)(env, env_params, runner_state, 
																							model_graph, model_state, model_other)
				else:
					a, b = jax.pmap(rlnmc_per_device, 
													in_axes = _in_axes,
													static_broadcasted_argnums = (0))(env, env_params, runner_state, 
																														model_graph, model_state, model_other)

			else:
				if NDEVICES == 1:
					a, b = jax.vmap(nmc_per_device, 
													in_axes = _in_axes)(env, env_params, runner_state)
				else:
					a, b = jax.pmap(nmc_per_device, 
													in_axes = _in_axes,
													static_broadcasted_argnums = (0))(env, env_params, runner_state)

			runner_state, traj_info = jax.tree.map(lambda x: x.block_until_ready(), (a, b))

			timer += time.time() - time_a

			traj_info["energies"] \
				= traj_info['energies'].swapaxes(0, 1).reshape(NUM_STEPS_PER_LOG, -1, K_ALL)
			traj_info["best_energies"] \
				= traj_info['best_energies'].swapaxes(0, 1).reshape(NUM_STEPS_PER_LOG, -1, K_ALL)
			traj_info["dtobest"] \
				= traj_info['dtobest'].swapaxes(0, 1).reshape(NUM_STEPS_PER_LOG, -1, K_ALL)

			final_energies = traj_info["energies"][-1]
			final_best_energies = traj_info["best_energies"][-1]
																
			if ground_states is not None:
				final_energies = final_energies - ground_states[None, :]
				final_best_energies = final_best_energies - ground_states[None, :]

			if track_diversity:
				time_diversity = 0
				time_div_a = time.time()

				best_energies_np = np.array(final_best_energies.swapaxes(0, 1))
				best_states_np = np.array(runner_state[0].best_s.reshape(NUM_REPLICAS, K_ALL, -1).swapaxes(0, 1))

				for i in range(K_ALL):
					for j in range(NUM_REPLICAS):
						if best_energies_np[i][j] <= approximation + jnp.finfo(jnp.float16).eps:
							#solutions for diversity across all replicas
							set_of_all_solutions[i].add(tuple(best_states_np[i][j]))

							if diversity_per_replica:
								#solutions for diversity in each replica
								set_of_solutions_pr[i][j].add(tuple(best_states_np[i][j]))

				time_diversity += time.time() - time_div_a
				

			LOG_SWEEP = (log_step+1)*NUM_STEPS_PER_LOG*STEP_MC_SWEEPS + INIT_MC_SWEEPS
			tts_results = tts_bootstrap(final_best_energies.swapaxes(0, 1), 
																	jnp.full((K_ALL,), LOG_SWEEP),
																	jnp.full((K_ALL,), approximation),
																	jax.random.key(0),
																	jnp.array(config["PERCENTILES"]), 
																	config["BOOTSTRAP"])

			all_tts_results.append(tts_results)

			metric = {}
			successes = {}
			bestes = {}

			if track_diversity:
				diversity = {r: {} for r in config["DIVERSITY_R"]}
				diversity["int"] = {}

				time_div_a = time.time()
				for name, solutions in zip(names, set_of_all_solutions):
					div_int, div_r = get_diversity(solutions, config["DIVERSITY_R"])

					#per r diversity
					for ri, r in enumerate(config["DIVERSITY_R"]):
						diversity[r][name] = div_r[ri]

					diversity["int"][name] = div_int

				for r in config["DIVERSITY_R"]:
					diversity_values = np.array(list(diversity[r].values()))
					
					metric[f"div_median/{r}"] = np.median(diversity_values)
					metric[f"div_mean/{r}"] = np.mean(diversity_values)
					metric[f"div_min/{r}"] = np.min(diversity_values)
					metric[f"div_max/{r}"] = np.max(diversity_values)
				
				metric[f"div_median/int"] = np.median(np.array(list(diversity["int"].values())))
				metric[f"div_mean/int"] = np.mean(np.array(list(diversity["int"].values())))
				metric[f"div_min/int"] = np.min(np.array(list(diversity["int"].values())))
				metric[f"div_max/int"] = np.max(np.array(list(diversity["int"].values())))

				time_diversity += time.time() - time_div_a

				num_solutions = {}
				for name, solutions in zip(names, set_of_all_solutions):
					num_solutions[name] = len(solutions)
				num_sol_values = np.array(list(num_solutions.values()))

				metric["num_sol_median"] = np.median(num_sol_values)
				metric["num_sol_mean"] = np.mean(num_sol_values)
				metric["num_sol_min"] = np.min(num_sol_values)
				metric["num_sol_max"] = np.max(num_sol_values)


			if config["LOG"]["WANDB"]:
				if K_ALL > 1:
					metric["pos"] = wandb.Histogram(tts_results["successes"]/(tts_results["successes"] + tts_results["failures"]))
					metric["tts_means"] = wandb.Histogram(tts_results["tts_means"])
				else:
					metric["pos"] = tts_results["successes"]/(tts_results["successes"] + tts_results["failures"])
					metric["tts_means"] = tts_results["tts_means"]

			else:
				if K_ALL > 1:
					metric["pos"] = jnp.histogram(tts_results["successes"]/(tts_results["successes"] + tts_results["failures"]))
					metric["tts_means"] = jnp.histogram(tts_results["tts_means"])
				else:
					metric["pos"] = tts_results["successes"]/(tts_results["successes"] + tts_results["failures"])
					metric["tts_means"] = tts_results["tts_means"]

			for name, suc_value, beste in zip(names, tts_results["successes"], final_best_energies.swapaxes(0, 1)):
				successes[name] = suc_value
				bestes[name] = beste

			if config["SAVE_RESULTS"]:
				for name in names:
					with open(os.path.join(suc_dir, name + "_suc.dat"), mode = 'a') as f:
						np.savetxt(f, np.array([LOG_SWEEP, successes[name]], dtype = int), 
											newline = " ", 
											fmt = "%i")

					with open(os.path.join(suc_dir, name + "_beste.dat"), mode = 'a') as f:
						np.savetxt(f, np.concatenate([[LOG_SWEEP], bestes[name].astype(int)], dtype = int), 
											newline = " ", 
											fmt = "%i")

					if track_diversity:
						with open(os.path.join(suc_dir, name + "_div.dat"), mode = 'a') as f:
							np.savetxt(f, np.concatenate([[LOG_SWEEP], 
																						[diversity[r][name] for r in config["DIVERSITY_R"]], 
																						[diversity["int"][name]]], dtype = float), 
												newline = " ", 
												fmt = "%.7g")

			for i, p in enumerate(config["PERCENTILES"]):
				metric[f"tts_{p}"] = tts_results["tts_p_means"][i]
				metric[f"s_tts_{p}"] = tts_results["tts_p_stds"][i]

			log_info = {"trajectory": {}, 
									"coarse": {"trajectory":{}}, 
									"fine": {}}

			log_info["trajectory"]["current_e"] = traj_info["energies"]
			log_info["trajectory"]["best_e"] 		= traj_info["best_energies"]
			log_info["trajectory"]["dtobest"] 	= traj_info["dtobest"]

			log_info['fine']["mc_stats"] 	 	 = jax.tree.map(lambda x: x.swapaxes(0, 1).reshape(NUM_STEPS_PER_LOG, -1), traj_info["mc_stats"])
			log_info['fine']["lbp_stats"] 	 = jax.tree.map(lambda x: x.swapaxes(0, 1).reshape(NUM_STEPS_PER_LOG, -1), traj_info["lbp_stats"])
			log_info['fine']["policy_stats"] = jax.tree.map(lambda x: x.swapaxes(0, 1).reshape(NUM_STEPS_PER_LOG, -1), traj_info["policy_stats"])

			if diversity_per_replica:
				div_array = np.empty((K_ALL, NUM_REPLICAS))
				num_sol_array = np.empty((K_ALL, NUM_REPLICAS))

				time_div_a = time.time()
				for i in range(K_ALL):
					for j in range(NUM_REPLICAS):
						div_int, _ = get_diversity(set_of_solutions_pr[i][j], config["DIVERSITY_R"])
						div_array[i, j] = div_int
						num_sol_array[i, j] = len(set_of_solutions_pr[i][j])
				time_diversity += time.time() - time_div_a
				
				log_info["coarse"]["trajectory"]["div"] = div_array
				log_info["coarse"]["trajectory"]["num_sol"] = num_sol_array

			all_log_info.append(log_info)

			other = {}

			other["total_seconds_elapsed"] = jnp.array(timer)
			other["seconds_per_log"] = jnp.array(timer - timer_old)
			
			if track_diversity:
				other["seconds_per_diversity"] = time_diversity

			other["total_sweeps_elapsed"] = jnp.array(LOG_SWEEP)
			other["sweeps_per_second"] = jnp.array(NUM_STEPS_PER_LOG*STEP_MC_SWEEPS)/other["seconds_per_log"]
			
			all_other.append(other)

			#here to log the stats
			if config["LOG"]["WANDB"]:
				bench_wandb_callback(LOG_SWEEP,
														 STEP_MC_SWEEPS,
														 log_info, 
														 metric,
														 other,
														 names, 
														 prefix)

			else:
				pass
		
		bench_results = runner_state, all_log_info, all_tts_results, all_other

		#here to save the stats
		if config["SAVE_ALL_DATA"]:
			#save the benchmarking statistics						
			with open(os.path.join(config["LOG"]["PATH"], 
														 config["LOG"]["PROJECT_NAME"],
														 config["LOG"]["NAME"] + ".pkl"), 'wb') as f:
				pickle.dump(bench_results, f)

		return bench_results

	return bench_nmc





##################################################################
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-config", "--config", 
											help = "config filename", 
											default = "config_bench_uf")
	args = parser.parse_args()

	with open(os.path.join("experiments", "configs", f"{args.config}.json"), 'r') as f:
		config = json.load(f)

	instance_ids = list(range(config["instances"][0], config["instances"][1] + 1))

	with open(os.path.join("problems", 
												 config["problem_class"], 
												 f"config.json"), 'r') as f:
		class_config = json.load(f)


	print("Running bechmarking of:")
	print(class_config["instance_name"])
	print("instances: ", instance_ids)
	print("num or replicas: ", config["bench"]["NUM_REPLICAS"])
	print("algorithm: ", config["bench"]["ALGO"])


	dtype = jnp.float32 if config["dtype"] == "float" else jnp.int32
	pmode = class_config["pmode"]

	instance_path = os.path.join("problems", 
																config["problem_class"],
																class_config["default_instances"])
	Jh_list = []
	gs_list = []
	for inst in instance_ids:
		instance_name = class_config["instance_name"] + f"_{inst}."

		if pmode in ["cnf", "wcnf"]:
			if class_config["file_type"] == "wcnf" and pmode == "cnf":
				raise RuntimeError("wrong pmode, wcnf required!")
				
			if os.path.isfile(os.path.join(instance_path, pmode + "_" + config["dtype"] + "_" + instance_name + "pkl")):
				with open(os.path.join(instance_path, pmode + "_" + config["dtype"] + "_" + instance_name + "pkl"), 'rb') as f:
					Jh, gs = pickle.load(f)

			else:
				_, _, cnf_file = wcnf_to_sparse_p(instance_path, instance_name + class_config["file_type"], config["dtype"],
																					pmode = pmode, 
																					no_weights = True if class_config["file_type"] == "cnf" else False,
																					gs_file = "gs_energies.txt")

				if pmode == "cnf":
					Jh = cnf_for_jax(cnf_file)
					
				else:
					wcnf_file = {}

					print(f"Running CNF/WCNF preprocessing of the problem {inst} for JAX...")
					if class_config["file_type"] == "cnf":
						for k, v in cnf_file.items():
							wcnf_file[k] = WCNF(jnp.ones((v.size//k), dtype=dtype) , jnp.array(v))
							
						Jh = wcnf_for_jax(wcnf_file, dtype=dtype)

					else:
						for k, v in cnf_file.items():
							wcnf_file[k] = WCNF(jnp.array(v[0]), jnp.array(v[1]))

						Jh = wcnf_for_jax(wcnf_file, dtype=dtype)

				gs = 0
				with open(os.path.join(instance_path, pmode + "_" + config["dtype"] + "_" + instance_name + "pkl"), 'wb') as f:
					pickle.dump((Jh, gs), f)

		else:
			if os.path.isfile(os.path.join(instance_path, pmode + "_" + config["dtype"] + "_" + instance_name + "pkl")):
				with open(os.path.join(instance_path, pmode + "_" + config["dtype"] + "_" + instance_name + "pkl"), 'rb') as f:
					Jh, gs = pickle.load(f)

			else:
				sJ, gs, _ = wcnf_to_sparse_p(instance_path, instance_name + class_config["file_type"], config["dtype"],
																		 pmode = pmode, 
																		 no_weights = True if class_config["file_type"] == "cnf" else False,
																		 gs_file = "gs_energies.txt")

				print(f"Running PUBO/PISING preprocessing of the problem {inst} for JAX...")
				Jh = sparse_ising_for_jax(sJ)
				with open(os.path.join(instance_path, pmode + "_" + config["dtype"] + "_" + instance_name + "pkl"), 'wb') as f:
					pickle.dump((Jh, gs), f)

		Jh_list.append(Jh)
		gs_list.append(jnp.array(gs, Jh.h.dtype))


	key = jax.random.key(config["seed"])

	if pmode == "pubo":
		pmode_idx = 0
	elif pmode == "pising":
		pmode_idx = 1
	elif pmode == "wcnf":
		pmode_idx = 3
	else:
		raise RuntimeError("Wrong pmode!")

	with open(os.path.join("modules", "policies", "nmc_ac.json"), 'r') as f:
		ac_config = json.load(f)

	env_params = EnvParams(
    max_steps_in_episode = class_config["NMC"]["max_steps_in_episode"],

    nmc_Nsw_init = class_config["NMC"]["nmc_Nsw_init"],
    nmc_Nsw_nbb = class_config["NMC"]["nmc_Nsw_nbb"],
    nmc_Nsw_bb = class_config["NMC"]["nmc_Nsw_bb"],
    nmc_Nsw_eq = class_config["NMC"]["nmc_Nsw_eq"],

    nmc_xT = class_config["NMC"]["nmc_xT"],
		
    Ti = class_config["NMC"]["Ti"],
    T = class_config["NMC"]["T"],
    Tf = class_config["NMC"]["Tf"],

    nmc_schedule = class_config["NMC"]["nmc_schedule"],

    nmc_init_annealing = class_config["NMC"]["nmc_init_annealing"],
    nmc_eq_annealing = class_config["NMC"]["nmc_eq_annealing"],
    nmc_rand_jump = class_config["NMC"]["nmc_rand_jump"],
    nmc_Ncycles = class_config["NMC"]["nmc_Ncycles"],

		lbp_beta = class_config["NMC"]["lbp_beta_x"]/class_config["NMC"]["T"], #the temperature of lbp is currently fixed
		
    lbp_num_iters = class_config["NMC"]["lbp_num_iters"],
    lbp_tolerance_m = class_config["NMC"]["lbp_tolerance_m"],
    lbp_tolerance_d = class_config["NMC"]["lbp_tolerance_d"],

    lbp_lambdas = jnp.array(class_config["NMC"]["lbp_lambdas"])
  )

	if config["bench"]["LOG"]["WANDB"]:
		wandb_run = wandb.init(
			project = config["bench"]["LOG"]["PROJECT_NAME"],
			config = {
				"problem_class": config["problem_class"],
				"dtype": config["dtype"],
				"track_stats": config["track_stats"],

				"instance_ids": instance_ids, 
				"pmode": pmode, 
				"BENCH": config["bench"],

				"class_config": class_config,
				
				"seed": config["seed"],

				"AC_PARAMS": ac_config if config["bench"]["ALGO"] == "rlnmc" else {}
			}
		)
		config["bench"]["LOG"]["NAME"] = wandb_run.id

	else:
		config["bench"]["LOG"]["NAME"] = config["bench"]["LOG"]["DEFAULT_NAME"]

	if config["bench"]["ALGO"] == "sa":
		(
			sminskey, #final state
			all_stats, #mc_stats at every log step
			all_tts_results, #tts_metric at every log step
			all_other #other stats at every log step
		) = \
			bench_sa(Jh_list, 
							 gs_list,
							 class_config["approximation"],
							 [str(i) for i in instance_ids], "bench", 
							 key, 
							 True, 
							 pmode_idx, 
							 config["bench"], 
							 devices = jax.devices()[:config["bench"]["NUM_DEVICES"]], 
							 succ_folder_name = args.suc_name)
		pass

	else:
		if config["bench"]["LOAD_PRETRAINED"] and (config["bench"]["ALGO"] == "rlnmc"):  
			if config["bench"]["LOAD_JSON"]:
				#load the models from a vanilla json dictionary
				load_dir = os.path.join("models_json", config["bench"]["JSON_NAME"] + ".json")

				with open(load_dir, 'r') as f:
					model_json = json.load(f)
				
				def convert_leaves_to_jax_array(d):
					if isinstance(d, dict):
							return {k: convert_leaves_to_jax_array(v) for k, v in d.items()}
					elif isinstance(d, list):
							return jnp.array(d, dtype = jnp.float32)
					else:
							return d 
				model_dict = convert_leaves_to_jax_array(model_json)
				
				abstract_model = nnx.eval_shape(lambda: FactorActorCritic(Jh_list[0], 
																																	pmode_idx, 
																																	**ac_config, 
																																	rngs = nnx.Rngs(0)))
				graphdef, model_state_pre, abstract_other = nnx.split(abstract_model, nnx.Param, ...)
				model_state_pre.replace_by_pure_dict(model_dict) #type: ignore

				print(f"Pretrained model {config['bench']['JSON_NAME']}, loaded via json")
				
			else:
				#load the models from a checkpointer
				load_dir = os.path.join(config["bench"]["LOAD_ORBAX"]["PATH"], 
																config["bench"]["LOAD_ORBAX"]["PROJECT_NAME"])

				checkpointer = ocp.StandardCheckpointer()
				abstract_model = nnx.eval_shape(lambda: FactorActorCritic(Jh_list[0], 
																																	pmode_idx, 
																																	**ac_config, 
																																	rngs = nnx.Rngs(0)))
				
				graphdef, abstract_state, abstract_other = nnx.split(abstract_model, nnx.Param, ...)
				model_state_pre = (
					checkpointer.restore(
						os.path.abspath(os.path.join(load_dir, 
																				config["bench"]["LOAD_ORBAX"]["PRETRAINED_ID"], 
																				config["bench"]["LOAD_ORBAX"]["MODEL_NAME"])), 
						abstract_state)
				)
				print(f"Pretrained model {config['bench']['LOAD_ORBAX']['PRETRAINED_ID']}, {config['bench']['LOAD_ORBAX']['MODEL_NAME']} loaded via orbax")
			
		else:
			model_state_pre = None

		config["bench"]["NMC"]["TOTAL_STEPS_PER_REPLICA"] = env_params.max_steps_in_episode
		
		bench_nmc = make_bench_nmc(Jh_list, 
															 gs_list, 
															 class_config["energy_scaler"],
															 class_config["approximation"],
															 [str(i) for i in instance_ids], "bench",
															 True, 
															 pmode_idx, 
															 config["bench"],
															 env_params,
															 devices = jax.devices()[:config["bench"]["NUM_DEVICES"]],
															 ac_config = ac_config, 
															 succ_folder_name = config["bench"]["LOG"]["NAME"])

		#run the benchmarking of the (RL)NMC													 
		runner_state, traj_info, tts_results, other = \
			bench_nmc(key, model_state_pre = model_state_pre)


