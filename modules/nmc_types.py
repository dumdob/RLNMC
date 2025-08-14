import jax
import jax.numpy as jnp
from typing import NamedTuple, List, Any

class WCNF(NamedTuple):     
  w: jax.Array
  indices: jax.Array

class Jsp(NamedTuple):      # Sparse interactions for pubo/p-ising/weighted-k-sat
  data: jax.Array
  indices: jax.Array

class Jhdata(NamedTuple):   # class for pubo/p-ising and weighted sat problems
  J: List[Jsp] 		          # sparse intraction matrices for energy computation
  h: jax.Array 			 		    # dense magnetic/bias field vectors
  Jat: List[Jsp]    				# batched interaction matrices for local field computation
  
class cnfdata(NamedTuple):  # a class for unweighted sat problems
  J: List[jax.Array] 		  	# cnf matrix
  h: jax.Array 			 		  	# single-variable clauses
  Jat: List[jax.Array]    	# clauses for each variable separately
  
class Minskey(NamedTuple):
	min_s: jax.Array
	min_e: jax.Array
	key: jax.Array
  
class Sminskey(NamedTuple):
	s: jax.Array
	min_s: jax.Array
	e: jax.Array
	min_e: jax.Array
	key: jax.Array
  
class Sminsnokey(NamedTuple):
	s: jax.Array
	min_s: jax.Array
	e: jax.Array
	min_e: jax.Array