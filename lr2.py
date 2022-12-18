import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition

f = open('lr2_text.txt', 'r')
text = f.read()

sentences = nltk.tokenize.sent_tokenize(text)

print('\n1. Word:\n')
cv = CountVectorizer(ngram_range=(1, 1))
x = cv.fit_transform(sentences)
print(cv.get_feature_names_out())
print("Array:")
print(x.toarray())

print('\n2. Bigram:\n')
cv = CountVectorizer(ngram_range=(2, 2), analyzer='char_wb')
x = cv.fit_transform(sentences)
print(cv.get_feature_names_out())
print("Array:")
print(x.toarray())

print('\n3. Phrase:\n')
cv = CountVectorizer(ngram_range=(2, 3))
x = cv.fit_transform(sentences)
print(cv.get_feature_names_out())
print("Array:")
print(x.toarray())

print("\nBefore decomposition:")
print(len(x.toarray()), len(x.toarray()[0]))

pca = decomposition.PCA()
pca.fit(x.toarray())
x = pca.transform(x.toarray())
# print(x)
print("\nAfter decomposition:")
print(len(x), len(x[0]))
