import numpy as np

# def selectRandomSeeds():
# 	"""Pick K random points as cluster centers called centroids.
# 	Args:
	    
# 	Returns:
# 	"""

def main():
	MATRIX = "data/matrix.txt"
	with open(MATRIX) as f:
		matrix = open(MATRIX, 'r').read()
    matrix = eval(matrix)
    print(matrix)

	# np.array([[hello[doc][term] for term in sorted(hello[doc])] for doc in sorted(hello)])


if __name__== "__main__":
	main()
