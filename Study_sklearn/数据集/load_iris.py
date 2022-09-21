from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
features = iris.data
target = iris.target
print(features.shape,target.shape)
