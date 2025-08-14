import jax
import jax.numpy as jnp
import wandb

from typing import Union, Dict, List

#Callback functions mainly designed for wandb logging of training and benchmarking

def get_stats(func, data: Union[Dict, jax.Array], axis = 0):
	stats = {}
	stats["one"]   = jax.tree.map(lambda x: func(jnp.take(x, 0, axis = axis)), data) # stats in a single replica
	stats["mins"]  = jax.tree.map(lambda x: func(jnp.min(x, axis = axis)), data)  
	stats["maxs"]  = jax.tree.map(lambda x: func(jnp.max(x, axis = axis)), data)

	stats["mean"] = jax.tree.map(lambda x: func(jnp.mean(x, axis = axis)), data)
	stats["stds"]    = jax.tree.map(lambda x: func(jnp.std(x, axis = axis)), data)

	stats["median"] = jax.tree.map(lambda x: func(jnp.median(x, axis = axis)), data)
	stats["perc20"]  = jax.tree.map(lambda x: func(jnp.percentile(x, q = 20, axis = axis)), data)
	stats["perc80"]  = jax.tree.map(lambda x: func(jnp.percentile(x, q = 80, axis = axis)), data)

	return stats

def train_wandb_callback(update, steps_per_update, info, loss_info, name):
	if wandb.run is None:
		raise RuntimeError("Wandb is not running!")

	wandb_stats = {}
	
	for i in range(steps_per_update):
		wandb_stats[f"{name}/trajectory"] = get_stats(jnp.mean, jax.tree.map(lambda x: x[i].squeeze(), info['trajectory']), 
																									axis = 0)
		wandb.log(data = wandb_stats, step = update + i + 1)
	
	wandb_stats[f"{name}/lbp_stats"]    = get_stats(jnp.mean, info['lbp_stats'], 		axis = 1)
	wandb_stats[f"{name}/mc_stats"]     = get_stats(jnp.mean, info['mc_stats'], 		axis = 1)
	wandb_stats[f"{name}/policy_stats"] = get_stats(jnp.mean, info['policy_stats'], axis = 1)
	
	if loss_info is not None:
		wandb_stats[f"train/total_loss"]   = get_stats(jnp.mean, loss_info[0], 		axis = 1)
		wandb_stats[f"train/actor_loss"]   = get_stats(jnp.mean, loss_info[1][0], axis = 1)
		wandb_stats[f"train/value_loss"]   = get_stats(jnp.mean, loss_info[1][1], axis = 1)
		wandb_stats[f"train/entropy_loss"] = get_stats(jnp.mean, loss_info[1][2], axis = 1)

	log_step = update + steps_per_update
	wandb.log(data = wandb_stats, step = log_step)


def bench_wandb_callback(sweep: int, 
												 sweeps_per_step: int,
												 info: Dict,
												 metric: Union[Dict, None],
												 other: Union[Dict, None],
												 names: List[str], 
												 prefix = "bench"):
	if wandb.run is None:
		raise RuntimeError("Wandb is not running!")

	wandb_stats = {}

	if 'fine' in info.keys():
		n_steps = jax.tree_leaves(info['fine'])[0].shape[0]

		for i in range(0, n_steps):
			wandb_stats[f"{prefix}/traj_stats"] = get_stats(jnp.mean, jax.tree.map(lambda x: x[i], info['fine']), axis = 0)

			for j, name in enumerate(names):
				wandb_stats[f"{name}/{prefix}/trajectory"] = get_stats(jnp.mean, jax.tree.map(lambda x: x[i, :, j], info['trajectory']), axis = 0)

			wandb.log(data = wandb_stats, step = sweep + (i - n_steps + 1)*sweeps_per_step)

	wandb_stats = {}
	if 'coarse' in info.keys():
		for i, name in enumerate(names):
			wandb_stats[f"{name}/{prefix}/coarse/trajectory"] = get_stats(jnp.mean, 
																																    jax.tree.map(lambda x: x[i], info['coarse']['trajectory']), axis = 0)
			
			if 'lbp_stats' in info['coarse'].keys():
				wandb_stats[f"{name}/{prefix}/coarse/lbp_stats"] = get_stats(jnp.mean, 
																																 		 jax.tree.map(lambda x: x[i], info['coarse']['lbp_stats']), axis = 0)

			if 'mc_stats' in info['coarse'].keys():
				wandb_stats[f"{name}/{prefix}/coarse/mc_stats"] = get_stats(jnp.mean, 
																																		jax.tree.map(lambda x: x[i], info['coarse']['mc_stats']), axis = 0)
				
			if 'policy_stats' in info['coarse'].keys():
				wandb_stats[f"{name}/{prefix}/coarse/policy_stats"] = get_stats(jnp.mean, 
																																				jax.tree.map(lambda x: x[i], info['coarse']['policy_stats']), axis = 0)

	if metric is not None:
		for key, value in metric.items():
			wandb_stats[f"metric/{key}"] = value

	if other is not None:
		for key, value in other.items():
			wandb_stats[f"other/{key}"] = value
		
	wandb.log(data = wandb_stats, step = sweep)
