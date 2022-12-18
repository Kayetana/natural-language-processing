from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

categories = ['comp.windows.x', 'rec.sport.baseball', 'sci.space', 'soc.religion.christian', 'talk.politics.guns']

newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
cv = CountVectorizer()
x = cv.fit_transform(newsgroups.data)

print('75/25')
X_train, X_test, Y_train, Y_test = train_test_split(x.toarray(), newsgroups.target, test_size=0.25)

print('Gaussian Naive Bayes')
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
print('Mean accuracy: ', gnb.score(X_test, Y_test))

print('C-Support Vector Classification')
svc = SVC(C=1.0, kernel='linear', degree=3, gamma="auto")
svc.fit(X_train, Y_train)
print('Mean accuracy: ', svc.score(X_test, Y_test))

print('----------\n50/50')
X_train, X_test, Y_train, Y_test = train_test_split(x.toarray(), newsgroups.target, test_size=0.5)

print('Gaussian Naive Bayes')
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
print('Mean accuracy: ', gnb.score(X_test, Y_test))

print('C-Support Vector Classification')
svc = SVC(C=1.0, kernel='linear', degree=3, gamma="auto")
svc.fit(X_train, Y_train)
print('Mean accuracy: ', svc.score(X_test, Y_test))
