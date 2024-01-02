import numpy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

matrix = np.loadtxt("output3.csv", delimiter = ",")
# matrix = matrix[:30,:30]
print(matrix)
heatmap = sns.heatmap(numpy.multiply(matrix,1), cmap=sns.color_palette('RdBu',100), )


plt.show()