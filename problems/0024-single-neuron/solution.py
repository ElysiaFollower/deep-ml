import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	
	# res = []
	# for x in features:
	# 	tmp = 0
	# 	for i in range(0, len(weights)):
	# 		tmp += x[i] * weights[i]
	# 	sig = 1 / (1 + math.exp(-(tmp+bias)))
	# 	res.append(round(sig, 4))
	
	# MSE = 0
	# i = 0
	# for pre in res:
	# 	MSE += math.pow((pre - labels[i]) , 2)
	# 	i+=1
	# MSE /= i
	# mse = MSE
	# probabilities = res

	import numpy as np 
	X = np.array(features)
	w = np.array(weights)
	y = np.array(labels)

	scores = X @ w + bias # X的最后一维和w的第0维匹配上，然后矩阵相乘
	prob = 1 / (1 + np.exp(-scores))
	probabilities = prob.tolist()

	MSE = np.mean( (prob-y)**2 )
	mse = MSE.tolist()

	return probabilities, mse