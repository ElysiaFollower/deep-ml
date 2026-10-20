from math import exp

def sigmoid(z: float) -> float:
	#Your code here
	result = 1 / (1 + exp(-z))
	return result