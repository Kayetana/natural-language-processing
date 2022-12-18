from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

n_clusters = 6
categories = ['comp.windows.x', 'rec.sport.baseball', 'sci.space', 'soc.religion.christian', 'talk.politics.guns']

newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
vectors = TfidfVectorizer()
x_vectors = vectors.fit_transform(newsgroups.data)

pca = PCA(n_components=2)
pca.fit(x_vectors.toarray())
x = pca.fit_transform(x_vectors.toarray())

gm = GaussianMixture(n_components=n_clusters, random_state=42)
gm_prediction = gm.fit_predict(x)

km = KMeans(n_clusters=n_clusters, random_state=42)
km_prediction = km.fit_predict(x)

km_centers = km.cluster_centers_
gm_centers = gm.means_

fig, axs = plt.subplots(2)
axs[0].scatter(x=x[:, 0], y=x[:, 1], c=km_prediction)
axs[0].scatter(x=km_centers[:, 0], y=km_centers[:, 1], marker='x', color="r")
axs[0].set_title('KMeans')
axs[1].scatter(x=x[:, 0], y=x[:, 1], c=gm_prediction)
axs[1].scatter(x=gm_centers[:, 0], y=gm_centers[:, 1, ], marker='x', color="r")
axs[1].set_title('Gaussian Mixture')
plt.savefig('lr4.png')