import json
import os
import sys
import time
import argparse

sys.path.append('.')

from modules.nmc_types import Jhdata
from modules.policies.nmc_policy import *
from modules.environment.nmcgym import *
from experiments.callbacks import *

from experiments.bench import make_bench_nmc

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Dict

from problems.readers import *

from flax import nnx
from flax.training import train_state

import optax
import distrax

from pprint import pprint
import pickle
import orbax.checkpoint as ocp

import wandb

class Transition(NamedTuple):
  done: jnp.ndarray
  action: jnp.ndarray
  value: jnp.ndarray
  reward: jnp.ndarray
  log_prob: jnp.ndarray
  obs: jnp.ndarray
  info: jnp.ndarray

def make_train(Jh: Jhdata, 
               ground_state: jax.Array,
               energy_scaler: float,
               approximation: float,
               track_stats: bool,
               pmode_idx: int, 
               config: Dict,
               ac_config: Dict,
               env_params: EnvParams):
  """
	Creates the training environment for a particular instance, and the recurrent PPO training logic function
  For each instances, many replicas are used to generate the RL data
	"""
  env = NMCgym(Jh = Jh, 
               nmc_schedule = env_params.nmc_schedule,
               energy_scaler = energy_scaler, 
               lbp_beta = env_params.lbp_beta, 
               track_stats = track_stats,
               bench_mode = False,
               approximation = approximation,
               pmode_idx = pmode_idx, 
               ground_state = ground_state)

  env = LogWrapper(env)

  def train(key: jax.Array, 
            xLR: float,
            graphdef, 
            params0, 
            other_variables):
    
    N = Jh.h.size

    tx = optax.chain(
      optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
      optax.adam(config["LR_end"] - (config["LR_end"] - config["LR_start"])*xLR)
    )

    class TrainState(train_state.TrainState):
      other_variables: nnx.State

    Tstate = TrainState.create(
      apply_fn = graphdef.apply,
      params = params0,
      other_variables = other_variables,
      tx = tx
    )
    
    key, subkey = jax.random.split(key) #jax random keys
    rngs = nnx.Rngs(subkey)             #nnx random keys

    #initialize the RNN hidden states
    h_init = model.gru_cell.initialize_carry((config["NUM_REPLICAS"], N, -1), rngs)
    if ac_config["global_gru"]:
      h_global_init = model.global_gru_cell.initialize_carry((config["NUM_REPLICAS"], 1, -1), rngs)
    else:
      h_global_init = None

    #Initialize the environment
    key, reset_pool_key = jax.random.split(key)
    reset_pool = env.reset_pool(key = reset_pool_key, 
                                pool_size = config["RESET_POOL"], 
                                min_start = True, #to reset in the best state of the annealing instead of the final one
                                params = env_params)
    reset_pool = jax.tree.map(lambda x: x[0], reset_pool)

    key, *reset_keys = jax.random.split(key, config["NUM_REPLICAS"] + 1)
    obsv, log_state = jax.vmap(env.reset, in_axes = (0, None, None))(jnp.stack(reset_keys), reset_pool, env_params)

    # TRAIN LOOP
    def _update_step(runner_state, unused):
      # COLLECT TRAJECTORIES
      def _env_step(runner_state, unused):
        Tstate, log_state, last_obs, last_done, h, h_global, key = runner_state
        key, subkey = jax.random.split(key)
    
        #local minimum observation
        x = jnp.stack((last_obs.s, last_obs.r), axis = 2)[None, :]
        extras = jnp.concatenate([last_obs.temp[:, :,  None], 
                                  last_obs.best_e[:, :,  None]], axis = 2)[None, :]

        (h, h_global, pi, value), (graphdef, new_state) = \
          Tstate.apply_fn(Tstate.params, Tstate.other_variables)(h, h_global, x, extras, last_done[None, :])

        pi_bernoulli = distrax.Bernoulli(logits = pi.squeeze(axis = 0))
        
        action = pi_bernoulli.sample(seed = subkey)
        log_prob = pi_bernoulli.log_prob(action).sum(axis = 1)
        
        # STEP ENV
        key, subkey = jax.random.split(key)
        step_keys = jax.random.split(subkey, config["NUM_REPLICAS"])

        obsv, log_state, reward, done, info = \
          jax.vmap(env.step, in_axes = (0, 0, 0, None, None))(step_keys, log_state, action, reset_pool, env_params)

        transition = Transition(
          last_done, action, value, reward, log_prob, last_obs, info
        )
        runner_state = (Tstate, log_state, obsv, done, h, h_global, key)
        return runner_state, transition

      #save initial recurrent states for later use
      h_initial = runner_state[-3]
      h_global_initial = runner_state[-2]

      runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS_PER_UPDATE"])

      Tstate, log_state, last_obs, last_done, h, h_global, key = runner_state

      # add extra sequence dimension
      x = jnp.stack((last_obs.s, last_obs.r), axis = 2)[None, :]
      extras = jnp.concatenate([last_obs.temp[:, :,  None], 
                                last_obs.best_e[:, :,  None]], axis = 2)[None, :]

      #computing the value function of the last state
      (_, _, _, last_val), (graphdef, new_state) = \
          Tstate.apply_fn(Tstate.params, 
                          Tstate.other_variables)(h, h_global, 
                                                  x, extras, 
                                                  last_done[None, :])
     
     
      # CALCULATE ADVANTAGE
      def _calculate_gae(traj_batch, last_val, last_done):
        def _get_advantages(carry, transition):
          gae, next_value, next_done = carry
          done, value, reward = (
            transition.done, 
            transition.value, 
            transition.reward
          )
          delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
          gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
          
          return (gae, value, done), gae
        _, advantages = jax.lax.scan(_get_advantages, 
                                     (jnp.zeros_like(last_val), last_val, last_done), 
                                     traj_batch, 
                                     reverse = True, 
                                     unroll = 16)
        return advantages, advantages + traj_batch.value
      
      advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

      # UPDATE NETWORK
      def _update_epoch(update_state, unused):
        def _update_minbatch(Tstate, batch_info):
          h_init, h_global_init, traj_batch, advantages, targets = batch_info

          def _loss_fn(params, other_variables, h_init, h_global_init, traj_batch, gae, targets):
            x = jnp.stack((traj_batch.obs.s, traj_batch.obs.r), axis = 3)
            # extras = jnp.concatenate([traj_batch.obs.e[:, :, :, None], 
            #                           traj_batch.obs.best_e[:, :, :, None], 
            #                           traj_batch.obs.dtobest[:, :, :, None]], axis = 3)
            extras = jnp.concatenate([traj_batch.obs.temp[:, :, :, None], 
                                      traj_batch.obs.best_e[:, :, :, None]], axis = 3)
            # extras = traj_batch.obs.best_e[:, :, :, None]

            # RERUN NETWORK
            (_, _, pi, value), (_, _) = \
              Tstate.apply_fn(params, other_variables)(h_init[0], 
                                                       h_global_init[0], 
                                                       x, extras,
                                                       traj_batch.done)

            pi_bernoulli = distrax.Bernoulli(logits = pi)

            log_probs0, log_probs1 = pi_bernoulli._log_probs_parameter()
            log_prob = (log_probs0*(1 - traj_batch.action) + log_probs1*traj_batch.action).sum(axis = 2)

            # log_prob = pi_bernoulli.log_prob(traj_batch.action) #doesn't work for some reason: float0 error

            # CALCULATE VALUE LOSS
            value_pred_clipped = traj_batch.value + (
              value - traj_batch.value
            ).clip(-config["CLIP_EPS_VF"], config["CLIP_EPS_VF"])

            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = value_losses_clipped.mean()

            # CALCULATE ACTOR LOSS
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = (
              jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
              )
              * gae
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

            entropy = pi_bernoulli.entropy().mean()

            total_loss = loss_actor + config["VF_COEF"]*value_loss - config["ENT_COEF"]*entropy
            
            return total_loss, (
              loss_actor, 
              config["VF_COEF"]*value_loss, 
              -config["ENT_COEF"]*entropy
            )
          grad_fn = jax.value_and_grad(_loss_fn, has_aux = True)
          total_loss, grads = grad_fn(Tstate.params, Tstate.other_variables,
                                      h_init, h_global_init, traj_batch, advantages, targets)
          
          Tstate = Tstate.apply_gradients(grads = grads)
          
          return Tstate, total_loss

        (
          Tstate,
          h_init,
          h_global_init,
          traj_batch,
          advantages,
          targets,
          key,
        ) = update_state

        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(subkey, config["NUM_REPLICAS"])
        batch = (h_init, h_global_init, traj_batch, advantages, targets)

        shuffled_batch = jax.tree_util.tree_map(
          lambda x: jnp.take(x, permutation, axis=1), batch
        )

        minibatches = jax.tree_util.tree_map(
          lambda x: jnp.swapaxes(
            jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])),
            1, 0,
          ),
          shuffled_batch,
        )

        Tstate, total_loss = jax.lax.scan(_update_minbatch, Tstate, minibatches)

        update_state = (
          Tstate,
          h_init,
          h_global_init,
          traj_batch,
          advantages,
          targets,
          key
        )
        return update_state, total_loss
      
      h_init = h_initial[None, :]
      h_global_init = h_global_initial[None, :]

      update_state = (
        Tstate,
        h_init,
        h_global_init,
        traj_batch,
        advantages,
        targets,
        key,
      )
      update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
      Tstate = update_state[0]

      key = update_state[-1]
      runner_state = (Tstate, log_state, last_obs, last_done, h, h_global, key)
      
      return runner_state, (traj_batch.info, loss_info)

    key, subkey = jax.random.split(key)
    runner_state = (
      Tstate,
      log_state,
      obsv,
      jnp.zeros((config["NUM_REPLICAS"]), dtype=bool),
      h_init,
      h_global_init,
      subkey
    )
    runner_state, running_info = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
    
    return_state = (
      runner_state[0].params,    #trained parameters of the model
      # runner_state[1],  #log_state
      # runner_state[2],  #obsverations
      runner_state[-1]  #random key
    )

    return return_state, {"traj_info": running_info[0], 
                          "loss_info": running_info[1]}

  return train








############################
if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("-config", "--config", 
                      help = "config filename", 
                      default = "config_train_uf")
  args = parser.parse_args()
  
  with open(os.path.join("experiments", "configs", f"{args.config}.json"), 'r') as f:
    config = json.load(f)

  instance_ids = list(range(config["instances"][0], config["instances"][1]+1))

  with open(os.path.join("problems", config["problem_class"], f"config.json"), 'r') as f:
    class_config = json.load(f)

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
          pickle.dump((Jh, 0), f)

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

  N = Jh_list[0].h.size
  assert np.all(np.array([Jh.h.size for Jh in Jh_list]) == N), "Different problem sizes given!"


  key = jax.random.key(config["seed"])
  eval_key = jax.random.key(config["seed"])

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

  train_config = config["train"]

  
  if config["train"]["LOG"]["WANDB"]:
    wandb_run = wandb.init(
      project = train_config["LOG"]["PROJECT_NAME"],

      config = {
        "problem_class": config["problem_class"],
        "dtype": config["dtype"],
        "track_stats": config["track_stats"],

        "instance_ids": instance_ids,

        "PRETRAIN": train_config,

        "class_config": class_config,
        
        "seed": config["seed"],

        "AC_PARAMS": ac_config
      }
    )
    LOG_NAME = wandb_run.id #type: ignore

  else:
    LOG_NAME = "local"

  #number of updates per each instance per repetition
  train_config["NUM_UPDATES"] = (
    train_config["TOTAL_TIMESTEPS_PER_INSTANCE"] // train_config["NUM_STEPS_PER_UPDATE"] // train_config["NUM_REPLICAS"]
  )

  TOTAL_REPETITIONS = train_config["TOTAL_REPETITIONS"]

  def outer_lr_schedule(count: int):
    frac = 1.0 - count/len(instance_ids)/TOTAL_REPETITIONS
    return frac

  timer = 0

  if env_params.nmc_rand_jump:
    STEP_MC_STEPS = (
      (1 + 1 + env_params.nmc_Nsw_eq)*env_params.nmc_Ncycles 
    )
  else:
    STEP_MC_STEPS = (
    ((env_params.nmc_Nsw_bb + env_params.nmc_Nsw_nbb)//2 + 
    env_params.nmc_Nsw_eq)*env_params.nmc_Ncycles  
    )

  count = 0

  save_dir = os.path.join(train_config["LOG"]["PATH"], 
                          train_config["LOG"]["PROJECT_NAME"])
  os.makedirs(save_dir, exist_ok = True) 

  checkpointer_model = ocp.StandardCheckpointer()

  model = FactorActorCritic(Jh_list[0], 
                            pmode_idx, 
                            **ac_config, 
                            rngs = nnx.Rngs(0))
  model_graph, \
  model_state, \
  model_other = nnx.split(model, nnx.Param, ...)

  if train_config["LOAD_PRETRAINED"]:  
    if train_config["LOAD_JSON"]:
      #load the models from a vanilla json dictionary
      load_dir = os.path.join("models_json", train_config["JSON_NAME"] + ".json")

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

      graphdef, model_state, abstract_other = nnx.split(abstract_model, nnx.Param, ...)
      model_state.replace_by_pure_dict(model_dict) #type: ignore  

      print(f"Pretrained model {train_config['JSON_NAME']}, loaded via json")

    else:
      #load the models from a checkpointer
      load_dir = os.path.join(train_config["LOAD_ORBAX"]["PATH"], 
                              train_config["LOAD_ORBAX"]["PROJECT_NAME"])
                              
      abstract_model = nnx.eval_shape(lambda: FactorActorCritic(Jh_list[0], 
                                                                pmode_idx, 
                                                                **ac_config, 
                                                                rngs = nnx.Rngs(0)))
      
      graphdef, abstract_state, abstract_other = nnx.split(abstract_model, nnx.Param, ...)
      model_state = checkpointer_model.restore(os.path.abspath(os.path.join(load_dir, 
                                               train_config["LOAD_ORBAX"]["PRETRAINED_ID"], 
                                               train_config["LOAD_ORBAX"]["MODEL_NAME"])), abstract_state)

      print(f"Pretrained {train_config['LOAD_ORBAX']['PRETRAINED_ID']}, {train_config['LOAD_ORBAX']['MODEL_NAME']} loaded via orbax")

  PATH_ERASED = False
  OLD_STATS_LOADED = False

  train_per_Jh = []
  model_per_Jh = []
  for Jh, gs in zip(Jh_list, gs_list):
    train_per_Jh.append(make_train(Jh, 
                                   gs,
                                   class_config["energy_scaler"],
                                   class_config["approximation"],
                                   config["track_stats"], 
                                   pmode_idx, 
                                   train_config, 
                                   ac_config, 
                                   env_params))
    
    model_per_Jh.append(FactorActorCritic(Jh, 
                                          pmode_idx, 
                                          **ac_config, 
                                          rngs = nnx.Rngs(0)))

  time_total = time.time()
  for rep_id in range(TOTAL_REPETITIONS):
    for inst_id, (Jh, gs) in enumerate(zip(Jh_list, gs_list)):
      metric_improved = False

      model = model_per_Jh[inst_id]
      model_graph, \
      _, \
      model_other = nnx.split(model, nnx.Param, ...)

      train = train_per_Jh[inst_id]

      #set the learning rate
      xLR = outer_lr_schedule(count)

      print(f"TRAINING {inst_id} REPETITION {rep_id} ...")
      timer_old = timer
      time_a = time.time()
      return_state, train_info = train(key, 
                                       xLR,
                                       model_graph, 
                                       model_state, 
                                       model_other)
      
      return_state, train_info = jax.tree.map(lambda x: x.block_until_ready(), (return_state, train_info))
      timer += time.time() - time_a
      print(f"TRAINING {inst_id} REPETITION {rep_id} COMPLETE")

      traj_info = train_info["traj_info"]
      loss_info = train_info["loss_info"]

      model_state = return_state[0]
      
      # log_stats = return_state[1]
      # observations = return_state[2]

      key = return_state[-1]
      print("current key: ", key)

      log_multiplier = train_config["TOTAL_TIMESTEPS_PER_INSTANCE"]//train_config["NUM_REPLICAS"]
      sweeps_elapsed = train_config["TOTAL_TIMESTEPS_PER_INSTANCE"]*STEP_MC_STEPS
      
      def log_stats(update_step, traj_info, loss_info):
        if config["train"]["LOG"]["WANDB"]:
          train_wandb_callback(update_step, 
                               train_config["NUM_STEPS_PER_UPDATE"], 
                               traj_info, 
                               loss_info, 
                               f"{inst_id}_{rep_id}")
          
        else:
          pass
        
      for update in range(train_config["NUM_UPDATES"]):
        log_stats(log_multiplier*count + update*train_config["NUM_STEPS_PER_UPDATE"], 
                  jax.tree.map(lambda x: x[update], traj_info), 
                  jax.tree.map(lambda x: x[update], loss_info))

      if config["train"]["LOG"]["WANDB"]:
        wandb_stats = {}
        wandb_stats[f"rl_stats/avg_returns"] = \
          jnp.mean(traj_info['returned_episode_returns'][traj_info['returned_episode']])
        wandb_stats[f"rl_stats/avg_ep_lengths"] = \
          jnp.mean(traj_info['returned_episode_lengths'][traj_info['returned_episode']])

        wandb_stats["total_seconds_elapsed"] = timer
        wandb_stats["seconds_per_instance"] = timer - timer_old
        wandb_stats["steps_per_second"] = log_multiplier*train_config["NUM_REPLICAS"]/(timer - timer_old)
        wandb_stats["sweeps_per_second"] = sweeps_elapsed/(timer - timer_old)
        
        wandb.log(wandb_stats, step = log_multiplier*(count + 1))

        wandb_stats["sweeps_per_instance"] = sweeps_elapsed

        print(f"RL STATS at step {log_multiplier*(count + 1)}:")
        pprint(wandb_stats)

      else:
        print("SECONDS PER INSTANCE", timer - timer_old)
        
        print("STEPS PER SECOND", log_multiplier*train_config["NUM_REPLICAS"]/(timer - timer_old))
        print("SWEEPS PER SECOND", sweeps_elapsed/(timer - timer_old))
        print("SWEEPS PER INSTANCE", sweeps_elapsed)

        print('EPISODE RETURNS:',)
        pprint(jnp.mean(traj_info['returned_episode_returns'][traj_info['returned_episode']]))

        print('EPISODE LENGTHS:')
        pprint(jnp.mean(traj_info['returned_episode_lengths'][traj_info['returned_episode']]))

        print("TOTAL SECONDS ELAPSED", timer)
      
      count += 1

      if PATH_ERASED == False:
        save_path = (
          ocp.test_utils.erase_and_create_empty(os.path.abspath(os.path.join(save_dir, LOG_NAME)))
        )
        PATH_ERASED = True

      checkpointer_model.save(save_path / f"model_{rep_id}_{inst_id}", model_state) #type: ignore
      checkpointer_model.wait_until_finished()
      print(f"Saved model at {time.time() - time_total} seconds")
          
    pass
  pass

  if PATH_ERASED == False:
    save_path = (
      ocp.test_utils.erase_and_create_empty(os.path.abspath(os.path.join(save_dir, LOG_NAME)))
    )

  #save the final state of the model
  checkpointer_model.save(save_path / "model_final", model_state) #type: ignore
  checkpointer_model.wait_until_finished()
  print(f"Saved FINAL model at {time.time() - time_total} seconds")

  print("FINISHED TRAINING")