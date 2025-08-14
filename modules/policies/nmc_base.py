import jax
import jax.numpy as jnp

@jax.jit
def threshold_backbones(r: jax.Array,
												cutoff: float) -> jax.Array:
	"""
	Simple thresholding of the variables for NMC: if m_i > cutoff, then True, else False

	Parameters:
	r (jax.Array): absolute variable magnetizations estimated by LBP
	cutoff (float): backbone threshold

	Returns:
	jax.Array: a boolean vector with backbone assignments
	"""
	bb = jnp.where(r > cutoff, True, False)

	return bb

@jax.jit
def random_backbones(m: jax.Array,
										 p: float, 
										 key) -> jax.Array:
	"""
	Random thresholding of the variables for NMC

	Parameters:
	m (jax.Array): variable magnetizations estimated by LBP
	p (float): backbone probability
	key: jax random key

	Returns:
	jax.Array: a boolean vector with backbone assignments
	"""
	bb = jnp.array(jax.random.bernoulli(key, p, m.shape), dtype=bool)

	return bb