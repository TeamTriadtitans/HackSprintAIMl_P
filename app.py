import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from io import BytesIO
import base64
from textblob import TextBlob
from starlette.requests import Request
from fastapi.templating import Jinja2Templates
#import wordcloud
from fastapi import FastAPI, HTTPException
import templates
import uvicorn
nltk.download('punkt')
nltk.download('stopwords')
import editSummary
from nltk.stem.porter import PorterStemmer
app=FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def home(request: Request):
    if request.method == "POST": 
        form = await request.form()
        if form["message"] and form["word_count"]:
            text = form["message"]
            summary = generate_summary(text)
            value = get_sentiment(summary)
            if value > 0:
                output = "Positive"
            elif value < 0:
                output = "Negative"
            else:
                output = "Neutral"
            #word_cloud = wordcloud(summary)
    return templates.TemplateResponse("index.html", {"request": request, "sumary": summary , "Analysis":output})


def read_article(text):
    sentences = sent_tokenize(text)
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
 
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
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

    sentences, words = read_article(text)

    sentence_similarity_matrix = build_similarity_matrix(words, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    # Step 5 - Get the top sentences as the summary
    for i in range(top_n):
      summarize_text.append("".join(ranked_sentences[i][1]))

    summary = " ".join(summarize_text)
    summary = summary.lower()
    summary = summary.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in summary if not word in set(all_stopwords)]
    review = ' '.join(review)
    #print(review)
    return review

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

text = """"
Nestled in the heart of the city, Gastronomique is a culinary gem that never fails to impress. From its elegant ambiance to its impeccable service and, most importantly, its delectable cuisine, Gastronomique sets the bar high for fine dining experiences.Walking into Gastronomique feels like stepping into a world of luxury and sophistication. The warm lighting, tasteful decor, and soft music create an inviting atmosphere perfect for a romantic dinner or a celebratory meal with friends and family. Whether you're seated by the cozy fireplace or at a table overlooking the bustling city streets, every corner of Gastronomique exudes charm and elegance."""
"""""
print(summary)

# File path
file_path = "output.txt"

# Open the file in write mode
with open(file_path, "w") as file:
    # Write the content to the file
    file.write(summary)

print("Content has been written to", file_path)

file_path = "output.txt"
# Read the content of the file
with open('output.txt', 'r') as file:
    data = file.read()

# Remove extra spaces from the content
cleaned_data = ' '.join(data.split())

# Write the modified content back to the file
with open('output.txt', 'w') as file:
    file.write(cleaned_data)

print("Extra spaces removed and content written back to the file.")
"""
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)