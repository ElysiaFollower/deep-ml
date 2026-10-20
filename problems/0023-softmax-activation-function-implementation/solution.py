import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    if len(scores) <= 0:
        return []


    # res = []
    # maxz = scores[0]
    # for z in scores:
    #     if z > maxz:
    #         maxz = z
    
    # for z in scores:
    #     res.append( math.exp(z - maxz) )
    
    # sumup = sum(res)
    # res = [ ez / sumup for ez in res]

    import numpy as np

    arr = np.array(scores)
    arr = np.exp(arr - arr.max())
    arr /= arr.sum()

    return arr.tolist()