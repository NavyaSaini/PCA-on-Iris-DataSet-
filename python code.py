import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn import datasets

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
from sklearn.preprocessing import StandardScaler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values

#visualise the data first to choose whether to center or standardise 
sns.scatterplot(df['sepal length'], df['sepal width'], hue = df['target'])
plt.title('Iris data by sepal length + width')
plt.legend(loc = 'lower right')
plt.show()

sns.scatterplot(df['petal length'], df['petal width'], hue = df['target'])
plt.title('Iris data by petal length + width')
plt.legend(loc = 'lower right')
plt.show()

# Standardizing the features because the features are independent as seen in plots
x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA

#1D representation

pcaone = PCA(n_components = 1)
pca1_vals = pcaone.fit_transform(x)
zeros = [0 for i in range(150)]
pc1_components = []
for i in pca1_vals:
    pc1_components.append(i[0])
plt.figure(figsize = (10,4))
sns.scatterplot(pc1_components, zeros, hue=df.target)
plt.title('Iris Data Set projected to first principal component')
plt.ylabel('Arbirary axis for visualisation')
plt.xlabel('Principal Component 1')
plt.show()

#in 2 PCs
pca = PCA(n_components=2)
components = pca.fit_transform(x)
pca2_df = pd.DataFrame(data = components, columns = ['pca1', 'pca2'])

sns.scatterplot(pca2_df.pca1, pca2_df.pca2, hue = df.target)
plt.ylabel('Principal Component 1')
plt.xlabel('Principal Component 2')
plt.title('Projection of Iris data into the first two PCs')
plt.show()

#Variance ratio
iris_pca = PCA(n_components=4)
iris_pca.fit_transform(x) #transform the axis of pca to fit the data
var_ratio = iris_pca.explained_variance_ratio_
var_ratio= pd.DataFrame(var_ratio).transpose()
var_ratio.columns = ['PC1', 'PC2', 'PC3', 'PC4']
var_ratio.index = ['Proportion of Variance']

plt.plot(['PC1','PC2','PC3','PC4'], [0.727705, 0.230305 , 0.036838 , 0.005152])
plt.title('Variance captured by each principal component')
plt.xlabel('Principal Component (PC)')
plt.ylabel('Variance captured')

