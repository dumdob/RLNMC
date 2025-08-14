import numpy as np
import scipy.sparse as sp
from typing import Tuple

#we don't use the "weighted" feature of the solver yet
from gurobi_optimods.mwis import maximum_weighted_independent_set

def build_graph_from_solution_set(set_of_solutions: set, 
																	R: float) -> Tuple[sp.coo_array, np.ndarray]:
	"""
	Construct a graph from solutions that is used to compute diversity

	Parameters:
		set_of_solutions (np.ndarray): A set of binary (True/False) vectors of solutions
		R (float): diversity threshold [0, 1]: 0 gives disconnected graph, 1 gives fully connected dense graph
	"""
	num_vertices = len(set_of_solutions)
	if num_vertices < 2:
		raise RuntimeError("Too few solutions given!")
	
	row = []
	col = []
	list_of_solutions = list(set_of_solutions)

	for i, si in enumerate(list_of_solutions):
		for j, sj in enumerate(list_of_solutions[i+1:]):
			# append the edges to the graph provided the threshold is satisfied
			if (np.array(si) != np.array(sj)).mean() <= R:
				row.append(i)
				col.append(i+1 + j)
	
	data = np.full((len(row), ), 1, dtype = np.int32)
	coo = sp.coo_array((data, (row, col)), shape = (num_vertices, num_vertices))

	#return the sparse graph and the weights of the vertices for the gurobi solver
	return coo, np.full((num_vertices, ), 1, dtype = np.int32)


def get_diversity(set_of_solutions: set, 
									R: np.ndarray) -> Tuple[float, list]:
	"""
	Construct a graph from solutions that is used to compute diversity

	Parameters:
		set_of_solutions (np.ndarray): A set of binary (True/False) vectors of solutions
		R (np.ndarray): an array of diversity thresholds

	Returns:
		The diversity integral value, as well as the [diverisity value for each R]
	"""
	D = []
	for r in R:
		if len(set_of_solutions) == 0:
			D.append(0)

		elif len(set_of_solutions) == 1:
			D.append(1)

		elif r == 0.0:
			#When r == 0, no edges in the graph: all solutions diverse
			D.append(len(set_of_solutions))

		elif r == 1.0:
			#When r == 1, the graph is fully connected: no diversity at all
			D.append(1)

		else:
			graph_adjacency, weights = build_graph_from_solution_set(set_of_solutions, r)

			mwis = maximum_weighted_independent_set(graph_adjacency, weights, verbose = False)

			D.append(mwis.f)
	
	if len(R) == 0:
		raise RuntimeError("Diversity R is not given!")

	elif len(R) == 1:
		return None, D
	
	else:
		D_integral = 0.0
		for i in range(len(D)):
			if i != len(D)-1:
				D_integral += (D[i] + D[i+1])*(R[i+1] - R[i])/2

		return D_integral/(R[-1] - R[0]), D