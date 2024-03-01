import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

nltk.download('punkt')
nltk.download('stopwords')

def read_article(text):
    # Split the text into sentences
    sentences = sent_tokenize(text)
    # Split the text into words
    words = [word_tokenize(sent.lower()) for sent in sentences]
    return sentences, words

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # Build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # Build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(text, top_n=2):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text and tokenize
    sentences, words = read_article(text)

    # Step 2 - Generate Similarity Matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(words, stop_words)

    # Step 3 - Rank sentences in similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the ranked sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    # Step 5 - Get the top sentences as the summary
    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentences[i][1]))

    # Step 6 - Combine the sentences to generate the summary
    summary = " ".join(summarize_text)
    return summary

# Example usage:
text = """
Nestled in the heart of the city, Gastronomique is a culinary gem that never fails to impress. From its elegant ambiance to its impeccable service and, most importantly, its delectable cuisine, Gastronomique sets the bar high for fine dining experiences.Walking into Gastronomique feels like stepping into a world of luxury and sophistication. The warm lighting, tasteful decor, and soft music create an inviting atmosphere perfect for a romantic dinner or a celebratory meal with friends and family. Whether you're seated by the cozy fireplace or at a table overlooking the bustling city streets, every corner of Gastronomique exudes charm and elegance.
"""

summary = generate_summary(text)
print(summary)
