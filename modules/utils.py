import jax
import jax.numpy as jnp

from typing import Dict, Any, Tuple

from functools import partial

@partial(jax.jit, static_argnames = ["bootstrap_size"])
def tts_bootstrap(energies: jax.Array, 
                  total_times: jax.Array,
                  approximation: jax.Array, 
                  key: jax.Array,
                  percentiles: jax.Array = jnp.array([10, 20, 50, 80, 90]),
                  bootstrap_size: int = 1000) -> Dict[Any, jax.Array]:
    """
    TTS bootstrapping from the parallel replica stats

    Parameters:
      energies (jax.Array): energies relative to the ground states for each instances
      total_times (jax.Array): lengths of runs for each instance
      approximation: approximation ratios for each instance
      key (jax.Array): jax random key
      bootstrap_size: size of the bootstrap samples
      percentiles: bootstrap tts percentiles
    """
    k = energies.shape[0]

    def get_success(e, approx):
      succ = (e <= approx + jnp.finfo(jnp.float16).eps).sum()
      fail = e.size - succ
      return succ, fail
    successes, failures = jax.vmap(get_success)(energies, approximation)
    
    keys = jax.random.split(key, bootstrap_size)
    def sample_tts(key: jax.Array):
      subkey1, subkey2 = jax.random.split(key)
      new_I = jax.random.choice(subkey1, jnp.arange(0, k), (k,), replace = True)
      keys = jax.random.split(subkey2, k)

      def get_tts(i, key):
        pos = jax.random.beta(key, successes[i] + 0.5, failures[i] + 0.5)
        r = jnp.log(0.01)/jnp.log(1.0 - jnp.clip(pos, 1e-15, 1.0 - 1e-15))
        tts = jnp.where(r < 1, total_times[i], total_times[i]*r)
        return tts
      return jax.vmap(get_tts)(new_I, keys)
      
    all_tts_samples = jax.vmap(sample_tts)(keys)

    tts_means = jnp.mean(all_tts_samples, axis = 0)
    tts_stds = jnp.std(all_tts_samples, axis = 0)

    tts_percentiles = jnp.array([jnp.percentile(all_tts_samples, p, axis = 1) for p in percentiles])
    tts_p_means = jnp.mean(tts_percentiles, axis = 1)
    tts_p_stds  = jnp.std(tts_percentiles, axis = 1)
        
    return {
       "successes": successes,
       "failures": failures,
       "tts_means": tts_means,
       "tts_stds": tts_stds,
       "tts_p_means": tts_p_means,
       "tts_p_stds": tts_p_stds
    }