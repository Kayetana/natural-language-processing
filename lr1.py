import re
import nltk
from polyglot.text import Word
from metaphone import doublemetaphone

t = """
Tesla is one of the worst performing stocks among major car makers and technology companies this year, as investors worry that Mr Musk's buyout of Twitter is diverting his attention.
On Wednesday the value of Tesla shares listed on the technology-heavy Nasdaq index in New York closed below $500bn for the first time since 2020.
At the end of last year the company was worth more than $1tn but its value has slumped in recent months.
Mr Musk completed the takeover of Twitter in October and since then has focused a significant amount of his time on the business.
Mr Musk sold billions of dollars worth of Tesla shares to help fund his purchase, which helped to push the shares down.
The Twitter deal was only completed after months of legal wrangling, and some have cited the distraction of the takeover as another factor behind Tesla's share price fall.
Investors have also been concerned that demand for the company's electric cars may slow, as the economy weakens, higher borrowing costs discourage buyers and other companies boost their electric vehicle offerings.
"""

text = re.sub(r'[^\w\s]', '', t)   # remove punctuation

print('1. Phonetic analysis')
sentence = nltk.sent_tokenize(text)[0]
phon = dict()
for j in nltk.word_tokenize(sentence):
    phon[j] = doublemetaphone(j)
print(phon)

print('----------\n\n2. Morphological analysis')
morph = dict()
for i in nltk.word_tokenize(text):
    morph[i] = Word(i, language='en').morphemes
print('Morphemes:\n', morph)
print('Parts of speech:\n', nltk.pos_tag(nltk.word_tokenize(text)))

print('----------\n\n3. Syntax analysis')
f = open('lr1_pattern.txt', 'r')
pattern = f.read()
sentences = nltk.sent_tokenize(text)
trees = []
for sent in sentences:
    tagged = nltk.pos_tag(nltk.word_tokenize(sent))
    Parser = nltk.RegexpParser(pattern)
    trees.append(Parser.parse(tagged))
print(trees)


sentences = nltk.sent_tokenize(text)
tagged = nltk.pos_tag(nltk.word_tokenize(sentences[0]))
Parser = nltk.RegexpParser(pattern)
tree = Parser.parse(tagged).draw()