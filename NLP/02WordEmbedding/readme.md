# NLP Concepts Quick Reference

A concise guide to fundamental NLP techniques with examples and Python snippets.

---

## 1. Named Entity Recognition (NER)

**Definition:** Identify and classify named entities in text (e.g., person, location, organization).

**Example:**

```text
Sentence: "Apple was founded by Steve Jobs in California."
Output:
- Apple → Organization
- Steve Jobs → Person
- California → Location
Python Example (using spaCy):

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple was founded by Steve Jobs in California.")

for ent in doc.ents:
    print(ent.text, "→", ent.label_)
2. Bag of Words (BoW)
Definition: Represents text by word frequency, ignoring order and grammar.

Example:

Sentence: "I love NLP and I love AI"
BoW representation:
- I: 2
- love: 2
- NLP: 1
- and: 1
- AI: 1
Python Example (using scikit-learn):

from sklearn.feature_extraction.text import CountVectorizer

corpus = ["I love NLP and I love AI"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
3. N-gram
Definition: A contiguous sequence of N words (or characters) to capture local context.

Example:

Sentence: "I love machine learning"
- Unigrams (1-gram): I, love, machine, learning
- Bigrams (2-gram): I love, love machine, machine learning
- Trigrams (3-gram): I love machine, love machine learning
Python Example (using scikit-learn):

from sklearn.feature_extraction.text import CountVectorizer

sentence = ["I love machine learning"]
vectorizer = CountVectorizer(ngram_range=(2,2))  # Bigrams
X = vectorizer.fit_transform(sentence)

print(vectorizer.get_feature_names_out())
4. TF-IDF (Term Frequency–Inverse Document Frequency)
Definition: Weights words based on their frequency in a document versus rarity across documents.

Example:

Documents:
D1: "I love AI"
D2: "I love NLP"

Observations:
- "love" → appears in both documents → low TF-IDF
- "AI" → appears in one document → high TF-IDF
Python Example (using scikit-learn):

from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["I love AI", "I love NLP"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
print(X.toarray())
5. Word2Vec
Definition: Converts words into dense vectors capturing semantic meaning.

Example:

- vector("king") − vector("man") + vector("woman") ≈ vector("queen")
- vector("dog") and vector("puppy") → vectors close together
Python Example (using gensim):

from gensim.models import Word2Vec

sentences = [["I", "love", "AI"], ["I", "love", "NLP"]]
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, workers=1)

print(model.wv['AI'])        # Vector for 'AI'
print(model.wv.most_similar('AI'))




| Method    | Considers Order | Captures Meaning | Output Type       |
|-----------|----------------|-----------------|------------------|
| BoW       | ❌ No           | ❌ No           | Sparse vector     |
| N-gram    | ✅ Partial      | ❌ No           | Sparse vector     |
| TF-IDF    | ❌ No           | ❌ No           | Weighted vector   |
| Word2Vec  | ✅ Yes          | ✅ Yes          | Dense vector      |
| NER       | ✅ Yes          | ✅ Yes          | Entity labels     |
```
