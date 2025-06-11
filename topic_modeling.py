!pip install --upgrade --force-reinstall numpy==1.23.5 gensim==4.3.1 nltk
!pip install -q spacy
!python -m spacy download en_core_web_sm
!pip install wordcloud
!pip install --upgrade --force-reinstall scipy==1.10.1
!pip install -q bertopic
!pip install -q sentence-transformers
!pip install pyLDAvis
!pip install top2vec

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def convert_rating(class_name):
    rating_dict = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
    for word, num in rating_dict.items():
        if word in class_name:
            return num
    return None

base_url = "https://books.toscrape.com/catalogue/"
page_url = "https://books.toscrape.com/catalogue/page-{}.html"

# Store scraped data
books = []

# Loop through all 50 pages (1000 books, 20 per page)
for page_num in range(1, 51):
    print(f"Scraping page {page_num}...")
    response = requests.get(page_url.format(page_num))
    soup = BeautifulSoup(response.text, 'html.parser')

    book_items = soup.find_all('article', class_='product_pod')

    for item in book_items:
        title = item.h3.a['title']
        price_text = item.find('p', class_='price_color').text.strip()
        price = float(price_text.encode('ascii', 'ignore').decode().replace('£', ''))
        rating_class = item.find('p', class_='star-rating')['class']
        rating = convert_rating(' '.join(rating_class))
        relative_url = item.h3.a['href']
        full_url = base_url + relative_url.replace('../../../', '')

        books.append({
            'title': title,
            'price': float(price),
            'rating': rating,
            'url': full_url
        })

# Create DataFrame
df_books = pd.DataFrame(books)
df_books.to_csv("books_listings.csv", index=False)
df_books.head()

def scrape_book_details(book_url):
    response = requests.get(book_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get product information table (as key-value pairs)
    table = soup.find('table', class_='table table-striped')
    rows = table.find_all('tr')
    product_info = {row.th.text.strip(): row.td.text.strip() for row in rows}

    # Get description
    desc_tag = soup.find('div', id='product_description')
    if desc_tag:
        description = desc_tag.find_next_sibling('p').text.strip()
    else:
        description = None

    # Get category from breadcrumb
    breadcrumb = soup.select('ul.breadcrumb li a')
    category = breadcrumb[2].text.strip() if len(breadcrumb) > 2 else None

    return {
        'upc': product_info.get('UPC'),
        'product_type': product_info.get('Product Type'),
        'price_excl_tax': product_info.get('Price (excl. tax)'),
        'price_incl_tax': product_info.get('Price (incl. tax)'),
        'availability_text': product_info.get('Availability'),
        'description': description,
        'category': category
    }

details = []

for i, row in df_books.iterrows():
    print(f"Scraping details for book {i+1}/{len(df_books)}...")
    info = scrape_book_details(row['url'])
    details.append(info)

# Merge into original dataframe
df_details = pd.DataFrame(details)
df_books_full = pd.concat([df_books, df_details], axis=1)

# Save
df_books_full.to_csv("books_full_details.csv", index=False)
df_books_full.head()

# Extract number in stock from "availability_text" (e.g., "In stock (22 available)")
import re
df_books_full['availability'] = df_books_full['availability_text'].apply(
    lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
)

# Reorder and keep useful columns
df_books_clean = df_books_full[[
    'title', 'category', 'rating', 'price',
    'availability', 'description'
]]

# Save cleaned version
df_books_clean.to_csv("books_cleaned.csv", index=False)
df_books_clean.head(10)

# Check for nulls and remove
df_books_clean[df_books_clean['description'].isnull()]
df_books_clean = df_books_clean[df_books_clean['description'].notnull()]
df_books_clean['description'].isnull().any()
len(df_books_clean)

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Book count per category (top 10)

top_categories = df_books_clean['category'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_categories.values, y=top_categories.index)
plt.title("Top 10 Categories by Book Count")
plt.xlabel("Number of Books")
plt.ylabel("Category")
plt.show()

# Price distribution (histogram)

plt.figure(figsize=(8, 5))
sns.histplot(df_books_clean['price'], bins=30, kde=True)
plt.title("Distribution of Book Prices")
plt.xlabel("Price (£)")
plt.ylabel("Frequency")
plt.show()

# Rating vs Price box plot

plt.figure(figsize=(8, 5))
sns.boxplot(x='rating', y='price', data=df_books_clean)
plt.title("Price Distribution by Star Rating")
plt.xlabel("Rating")
plt.ylabel("Price (£)")
plt.show()

# Install and load spaCy
import spacy
from collections import Counter

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

# Combine all book descriptions into one string
descriptions = df_books_clean['description'].dropna().str.lower().str.cat(sep=' ')

# Process the full text with spaCy
doc = nlp(descriptions)

# Filter tokens: remove stopwords, adverbs, punctuation, short words
filtered_tokens = [
    token.text for token in doc
    if not token.is_stop and token.pos_ != 'ADV' and token.is_alpha and len(token.text) > 2
]

# Count word frequencies
word_freq = Counter(filtered_tokens)
common_words = word_freq.most_common(20)
common_words

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all filtered tokens into a single string
text_for_wordcloud = ' '.join(filtered_tokens)

# Generate word cloud
wordcloud = WordCloud(
    width=800, height=400,
    background_color='white',
    max_words=100
).generate(text_for_wordcloud)

# Display the word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Words in Book Descriptions", fontsize=16)
plt.show()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from nltk.corpus import stopwords
import re
import nltk
nltk.download('stopwords')

def remove_stops(text, stops):
    text = re.sub(r"\.\.\.more", "", text)
    words = text.split()
    final = []
    for word in words:
        if word not in stops:
            final.append(word)
    final = " ".join(final)
    final = final.translate(str.maketrans("", "", string.punctuation))
    final = "".join([i for i in final if not i.isdigit()])
    while "  " in final:
        final = final.replace("  ", " ")
    return (final)

def clean_docs(docs):
    stops = stopwords.words("english")
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    stops = stops + months
    final = []
    for doc in docs:
        clean_doc = remove_stops(doc, stops)
        final.append(clean_doc)
    return final

descriptions = df_books_clean['description']
titles = df_books_clean['title']

print(descriptions[0])
print(titles[0])

cleaned_docs = clean_docs(descriptions)
print(cleaned_docs[0])

vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=100,
    max_df=0.8,
    min_df=5,
    ngram_range=(1, 3),
    stop_words="english"
)

vectors = vectorizer.fit_transform(cleaned_docs)

feature_names = vectorizer.get_feature_names_out()

dense = vectors.todense()
denselist = dense.tolist()

all_keywords = []

for description in denselist:
    x = 0
    keywords = []
    for word in description:
        if word > 0:
            keywords.append(feature_names[x])
        x = x + 1
    all_keywords.append(keywords)

print(descriptions[0])
print(all_keywords[0])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Try different k values
range_k = range(2, 21)
silhouette_scores = []

for k in range_k:
    kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=10, random_state=42)
    kmeans.fit(vectors)
    labels = kmeans.labels_
    score = silhouette_score(vectors, labels)
    silhouette_scores.append(score)

# Plotting silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(range_k, silhouette_scores, marker='o')
plt.title("Silhouette Score for Different k Values")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

true_k = 20

model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)
model.fit(vectors)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

# Show top 10 terms per cluster
for i in range(true_k):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"Cluster {i}: {', '.join(top_terms)}")

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy
import spacy
from nltk.corpus import stopwords

# vis
import pyLDAvis
import pyLDAvis.gensim

stopwords = set(stopwords.words('english'))
print(stopwords)

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return texts_out

# Apply lemmatization function to DataFrame's 'description' column
lemmatized_descriptions = lemmatization(df_books_clean["description"])

print(lemmatized_descriptions[0])

def gen_words(texts):
  final = []
  for text in texts:
    new = gensim.utils.simple_preprocess(text, deacc=True)
    final.append(new)
  return (final)

data_words = gen_words(lemmatized_descriptions)

print(data_words[0])

# Bigrams and Trigrams

bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)

bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)

def make_bigrams(texts):
  return(bigram[doc] for doc in texts)

def make_trigrams(texts):
  return (trigram[bigram[doc]] for doc in texts)

data_bigrams = list(make_bigrams(data_words))
data_bigrams_trigrams = list(make_trigrams(data_bigrams))

print(data_bigrams_trigrams[0])

# TF-IDF Removal

from gensim.models import TfidfModel

id2word = corpora.Dictionary(data_bigrams_trigrams)

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts]
print(corpus[0][0:10])

tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.03
words = []
words_missing_in_tfidf = []
for i in range(0, len(corpus)):
  bow = corpus[i]
  low_value_words = [] # Reinitialize to be safe
  tfidf_ids = [id for id, value in tfidf[bow]]
  bow_ids = [id for id, value in bow]
  low_value_words = [id for id, value in tfidf[bow] if value < low_value]
  drops = low_value_words+words_missing_in_tfidf
  for item in drops:
    words.append(id2word[item])
  words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # Words with tfidf score 0 will be missing

  new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
  corpus[i] = new_bow

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=30,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto')

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
vis

docs = df_books_clean.description.tolist()
docs[0]

from top2vec import Top2Vec

model = Top2Vec(docs)

topic_sizes, topic_nums = model.get_topic_sizes()
print(topic_sizes)

print(topic_nums)

topic_words, word_scores, topic_nums = model.get_topics(1)

for words, scores, num in zip(topic_words, word_scores, topic_nums):
    print(num)
    print(f"Words: {words}")

documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=0, num_docs=10)

for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("----------------------------------------")
    print(doc)
    print()
