import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	import numpy as np
	res = []
	for x in features:
		tmp = 0
		for i in range(0, len(weights)):
			tmp += x[i] * weights[i]
		sig = 1 / (1 + math.exp(-(tmp+bias)))
		res.append(round(sig, 4))
	
	MSE = 0
	i = 0
	for pre in res:
		MSE += math.pow((pre - labels[i]) , 2)
		i+=1
	MSE /= i
	mse = MSE
	probabilities = res

	return probabilities, mse