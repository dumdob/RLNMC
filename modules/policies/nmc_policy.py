import json
import os
import sys
import time

sys.path.append('.')
from modules.nmc_types import Jhdata

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Tuple, Union, List

from flax import nnx
import distrax

class FactorNnnz(nnx.Variable): pass
class FactorIndices(nnx.Variable): pass

def get_factor_data(Jh: Jhdata, 
										pmode_idx: int):
	"""
	Parameters:
		Jh (Jhdata): a factor graph of the problem (only highest order so far)
		pmode_idx: index of the pmode: pubo/pising/cnf/wcnf
	"""

	if pmode_idx in [0, 1]:
		indices = Jh.J[-1].indices
	else:
		#convert cnf to simple indices in factors
		indices = jnp.abs(Jh.J[-1].indices)-1

	#total number of variables (nodes)
	N = Jh.h.size

	#number of variable appearances in factors of every order
	Nnnz = jnp.sum((Jh.Jat[-1].data != 0), axis = 1)

	return N, FactorNnnz(Nnnz), FactorIndices(indices)


class FactorActorCritic(nnx.Module):
	def __init__(self, 
							 Jh: Jhdata,
							 pmode_idx: int,
							 subgraphs: int = 1,
							 *,
							 din: int, 
							 dmsgs: int, 
							 att_heads: int, 
							 dextra: int,
							 dout: int,
							 rngs: nnx.Rngs, 
							 global_gru: bool = False):
		
		self.N, self.Nnnz, self.indices = get_factor_data(Jh, pmode_idx)
		self.subgraphs = subgraphs
		
		self.din = din # typicaly din = 2: magnetization and state
		self.dmsgs = dmsgs
		self.att_heads = att_heads
	
		#extra information provided to the policy (energy, distance, etc.)
		self.dextra = dextra
		self.dout = dout

		self.global_gru = global_gru

		#GRU layers
		self.gru_cell = nnx.GRUCell(din, dmsgs, rngs = rngs)

		#multihead self-attention (only highest order supported so far)
		self.attention = nnx.MultiHeadAttention(num_heads = att_heads, 
																						in_features = dmsgs,  
																						qkv_features = dmsgs,
																						out_features = dmsgs,
																						decode = False,
																						rngs = rngs)

		if self.global_gru:
		#global GRU/LSTM recurrence is used to create backbone schedules
			self.global_gru_cell = nnx.GRUCell(dmsgs + dextra, dout, rngs = rngs)

			self.output = nnx.Sequential(
				nnx.Linear(dout + dmsgs, dout, rngs = rngs),
				nnx.relu, 
				nnx.Linear(dout, 1, rngs = rngs)
			)

			self.output_value = nnx.Sequential(
				nnx.Linear(dout, 1, rngs = rngs)
			)

		else:
			self.output = nnx.Sequential(
				nnx.Linear(dmsgs + dextra, dout, rngs = rngs), 
				nnx.relu, 
				nnx.Linear(dout, 1, rngs = rngs)
			)

			self.output_value = nnx.Sequential(
				nnx.Linear(dmsgs + dextra, 1, rngs = rngs)
			)


	def __call__(self, 
							 h: jax.Array, 
							 h_global: jax.Array,
							 x: jax.Array, 
							 extra: jax.Array,
							 dones: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
		"""
		Parameters:
			self
			h (jax.Array): recurrent hidden state
			x (jax.Array): local minimum input: state/magnetization
			extra (jax.Array): extra info: energy, distance, etc.
			h_global (jax.Array): optional global hidden state to create backbone schedules
		"""
		seq_length = x.shape[0]
		batch_size = x.shape[1]
		K_all = extra.shape[2]
		N0 = self.N//K_all

		#apply rnn to every node input (memory of state/magnetization)
		def gru_scan_fn_with_reset(carry, cell, ins_reset):
			ins, reset = ins_reset
			rnn_state = jnp.where(reset[:, None, None], 
														cell.initialize_carry(carry.shape, nnx.Rngs(0)), 
														carry)
			return cell(rnn_state, ins)

		h, hs = nnx.scan(gru_scan_fn_with_reset, 
										 in_axes = (nnx.Carry, None, 0), 
										 out_axes = (nnx.Carry, 0))(h, self.gru_cell, (x, dones))

		#apply self attention in each factor of the factor graph (correlations analysis)
		h_to_att = jnp.take(hs, self.indices.value, axis = 2)
		a = self.attention(h_to_att)
		
		a = a.reshape(seq_length, batch_size, -1, a.shape[-1])

		j_segment_sum = jax.jit(jax.ops.segment_sum, static_argnames = 'num_segments')
		def segment_sum_per_batch(a):
			return jax.lax.map(lambda x: j_segment_sum(x, self.indices.flatten(), num_segments = self.N), a)		
		y = jax.vmap(segment_sum_per_batch, in_axes = 1, out_axes = 1)(a)
		
		y /= self.Nnnz[None, None, :, None] #normalize the segment sum
		
		y_per_instance = y.reshape((seq_length, batch_size, K_all, -1, y.shape[-1]))
		y_global = jnp.mean(y_per_instance, axis = 3)
		if self.dextra > 0:
			z_global = jnp.concatenate((y_global, extra), axis = 3)
		else:
			z_global = y_global
		
		if not self.global_gru:
			#simple MLP output that uses extra information
			if self.dextra > 0:
				extra_tiled = jnp.tile(extra[:, :, :, None, :], (1, 1, 1, N0, 1))
				z = jnp.concatenate((y_per_instance, extra_tiled), axis = 4).reshape(seq_length, batch_size, self.N, -1)
			else:
				z = y

			#the sum over the instances (during training only one instance is used at a time)
			value = self.output_value(z_global).sum(axis = 2)

		else:
			#global GRU output that uses extra information
			h_global, hs_global =\
				nnx.scan(gru_scan_fn_with_reset, 
								 in_axes = (nnx.Carry, None, 0), 
								 out_axes = (nnx.Carry, 0))(h_global, 
																						self.global_gru_cell, 
																						(z_global, dones))
			
			hs_global_tiled = \
				jnp.tile(hs_global[:, :, :, None, :], (1, 1, 1, N0, 1)).reshape(seq_length, batch_size, self.N, -1)																			 
			z = jnp.concatenate((y, hs_global_tiled), axis = 3)

			#the sum over the instances (during training only one instance is used at a time)
			value = self.output_value(hs_global).sum(axis = 2)

		logit = self.output(z).squeeze(axis = 3)

		return h, h_global, logit, value.squeeze()

if __name__=='__main__':
	pass
