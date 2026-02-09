#Q1:Bag of words models represent text as sparse counts of exact words, prioritizing simplicity whereas embedding-based representations like Word2Vec or Transformers map words to dense vectors that capture complex semantic relationships and context. A key trade-off involves interpretability versus robustness: BoW allows for easy inspection of which words drive a model's decisions but fails to recognize synonyms, while embeddings are robust to linguistic variation but function as black boxes. Consequently, a social scientist might prefer BoW for a longitudinal study tracking specific policy keywords to ensure exact measurement validity. Conversely, they would choose embeddings for tasks like detecting hate speech in social media, where capturing sarcasm, slang, and implied context is essential for accurate classification.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from gensim.models import Word2Vec

df = pd.read_csv('Penn State/Spring 2026/SoDA 501/Week 4/data_raw/week_movie_corpus.csv')

vectorizer = CountVectorizer(lowercase=True,stop_words="english",token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",min_df=5)

X_counts = vectorizer.fit_transform(df["text"].astype(str))
vocab = vectorizer.get_feature_names_out()

lda = LatentDirichletAllocation(n_components=6,random_state=19880106,learning_method="batch")
lda.fit(X_counts)

topic_word = lda.components_
n_top_words = 10

for k in range(6):
    top_word_idx = topic_word[k].argsort()[::-1][:n_top_words]
    words = vocab[top_word_idx]
    print(f"Topic {k}: {', '.join(words)}")

doc_topic_dist = lda.transform(X_counts)
df["dominant_topic"] = doc_topic_dist.argmax(axis=1)

topic_counts = df["dominant_topic"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(topic_counts.index.astype(str), topic_counts.values, color='skyblue', edgecolor='black')
plt.title("Number of Documents per Dominant Topic (LDA)")
plt.xlabel("Topic Label")
plt.ylabel("Document Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Penn State/Spring 2026/SoDA 501/Week 4/figures/lda_topic_distribution.png", dpi=200)
plt.show()

#Topic 5 represents a Family Drama or Romance theme, characterized by terms like love, marriage, family, and son, while Topic 1 captures Action or Thriller elements through words like team, police, and plane. The quality of these topics highlights the importance of preprocessing; for example, the recurring appearance of the word film in Topics 2, 3, and 4 indicates that the standard English stopword list might be insufficient, and adding film to a custom stopword list would have reduced this noise. Additionally, the prevalence of common names, e.g., John, Michael, suggests that without aggressive filtering, such as removing proper nouns or lowering max_df, character names can crowd out more descriptive thematic words.


tokenized_docs = [re.findall(r"(?u)\b\w+\b", str(text).lower()) for text in df["text"]]

w2v_model = Word2Vec(sentences=tokenized_docs,vector_size=100,window=5,min_count=2,workers=4,seed=123)

doc_vectors = []
for tokens in tokenized_docs:
    valid_vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if len(valid_vectors) > 0:
        doc_vectors.append(np.mean(valid_vectors, axis=0))
    else:
        doc_vectors.append(np.zeros(w2v_model.vector_size))

X = np.array(doc_vectors)
y = df["y_outcome"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

ridge = Ridge(alpha=1.0, random_state=123)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2:  {r2:.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predictions')
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Fit Line')
plt.title("Word2Vec + Ridge: Actual vs. Predicted")
plt.xlabel("Actual Outcome (Binary: 0 or 1)")
plt.ylabel("Predicted Outcome (Continuous)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Penn State/Spring 2026/SoDA 501/Week 4/figures/word2vec_regression_actual_vs_predicted.png", dpi=200)
plt.show()

#Regressing an outcome on text embeddings involves using the dense vector representations of documents—created by averaging Word2Vec word vectors—as input features to predict a target variable, such as whether a movie is an Action film. The low R2 of 0.0172 in the results indicates that this specific model explains almost none of the variance in the outcome, essentially performing no better than guessing the average. A major limitation of this approach is the loss of syntax and context; by simply averaging word vectors to create a document embedding, the model discards word order and negation, which likely contributed to the poor predictive performance. Additionally, this method suffers from poor interpretability, as the embedding dimensions are abstract numbers rather than explicit words, making it impossible to explain exactly why the model classified a movie as a specific genre compared to a simple word-count model.
