# Fundamentals of Text Processing

An educational course on natural language processing and machine learning methods for working with textual data. Built around hands-on practice with Python, scikit-learn, PyTorch, and gensim: from basic parsing and TF-IDF to recurrent neural networks and generative models.

## What the course is about

Modern text processing relies on methods that turn sequences of characters into numerical representations and teach models to extract meaning from them — from individual words and n-grams to whole documents and generated texts. The course walks through the key stages of working with such models: data collection and cleaning, vectorization, training, and evaluation.

**Data.** A text in a model is a sequence of tokens, and a corpus is a collection of documents. Preprocessing (lowercasing, removing links, hashtags and punctuation, tokenization) stabilizes training, while a train/test split with class balance preserved (`stratify`) ensures correct evaluation. Sources range from scraped website datasets to ready-made corpora (e.g., positive.csv / negative.csv with reviews).

**Vectorization.** Classical models cannot work with raw text, so it is converted into features. TF-IDF weights words by their importance in a document and rarity in the corpus, while n-grams (n = 1..5) add information about word order and collocations. Studying the effect of the n parameter shows how growing context affects classification quality.

**Embeddings.** Word2Vec and FastText represent words as dense vectors in which proximity reflects semantics. FastText additionally builds vectors from character n-grams, which makes it possible to obtain representations for rare and unknown words. Comparing the models demonstrates that semantic similarity does not always match intuition: synonyms may end up farther apart than antonyms, and polysemous words attract neighbors from different senses.

**Classical models.** Logistic regression, linear SVM, and Multinomial Naive Bayes are compared on the same TF-IDF matrix. Different algorithms balance precision and recall differently, and the effect of stop words and text cleaning shows how important preprocessing is for the final metrics.

**Recurrent networks.** LSTM accounts for word order and context, which matters for text classification and generation. Regularization (dropout, weight decay, limiting vocabulary and sequence length) combats overfitting: without it, the model either degrades or memorizes the training set. Poetry generation with LSTM shows how data quality and network structure affect the coherence and meaningfulness of the output.

**Architectures.** From a basic unidirectional LSTM to a bidirectional multi-layer one with regularization — each step complicates the model and balances expressiveness against the risk of overfitting.

## Course structure

| # | Topic | Key tools |
|---|-------|-----------|
| 1 | Data parsing and corpus preparation | `requests`, `BeautifulSoup`, `csv` |
| 2 | TF-IDF, n-grams, and classifier comparison | `scikit-learn` (LogisticRegression, LinearSVC, MultinomialNB) |
| 3 | Word embeddings: Word2Vec and FastText | `gensim` |
| 4 | LSTM for text classification | `PyTorch`, `nn.LSTM`, `DataLoader` |
| 5 | Text generation and combating training degradation | `TensorFlow/Keras`, regularization, sparse representations |