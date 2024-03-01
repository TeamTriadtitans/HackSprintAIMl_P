import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def read_file_to_string(file_path):
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            return file_content
    except FileNotFoundError:
        print("File not found!")
        return None

file_path = "D:\semtiment Anal\output.txt"
summary = read_file_to_string(file_path)

#summary = summary.split()
#summary= ' '.join(''.join(word.split()) for word in summary)
#summary = summary.replace(" ","")
#summary=summary.replace(" ","")
#print(summary)
#print(summary)
#review = re.sub('[^a-zA-Z]', ' ', summary)


summary = summary.lower()
summary = summary.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
review = [ps.stem(word) for word in summary if not word in set(all_stopwords)]
review = ' '.join(review)
print(review)