import random
import pandas as pd
from gensim import downloader as api
from csv import reader
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from scipy.sparse import data
import Task1

# First model: glove-wiki-gigaword-300 (C1-E1)
# Second model: fasttext-wiki-news-subwords-300 (C2-E2)
# Third model: glove-twitter-50 (C3-E3)
# Fourth model: glove-twitter-200 (C4-E4)
#note: C1 /= C2 and C3 = C4 and  E1 = E2 and E3 /= E4.

print("importing models:")
# Load the models and make them go threw task1 calculations
glove_wiki_gigaword_300 = Task1.bagOfWords(corpora="glove-wiki-gigaword-300")
fasttext_wiki_news_subwords_300 = Task1.bagOfWords(corpora="fasttext-wiki-news-subwords-300")
glove_twitter_50 = Task1.bagOfWords(corpora="glove-twitter-200")
glove_twitter_200 = Task1.bagOfWords(corpora="glove-twitter-25")

# calculate accuracy for
accuracy_rate= pd.read_csv('Crowdsourced Gold-Standard for MP3', index_col=3)

human_standards = accuracy_rate["Sucess Rate"].tolist()
specify_human_standards = {x for x in human_standards if x == x}
iterate_human_standards = [x[:4] for x in specify_human_standards]
human_standards_fl = [float(i) for i in iterate_human_standards] # display as float for accuracy

# create a random baseline
random_baseline = human_standards_fl
for i in range(len(random_baseline)): #iterate throw random base-line
    random_baseline[i] = round(random.uniform(0, 100), 1)
# calculate the values of
humanStandards = round((sum(human_standards_fl) / len(human_standards_fl)),2) # Calculates the average for the Crowd Sourced Gold Standard
randomBaseline = round((sum(random_baseline) / len(random_baseline)), 2)  # Calculates the average for the random baseline

x = ["glove-wiki-gigaword-300", "fasttext-wiki-news-subwords-300", "glove-twitter-200", "glove-twitter-25",
     "Crowd Sourced Gold Standard", "Random Baseline"]
y = [glove_wiki_gigaword_300, fasttext_wiki_news_subwords_300, glove_twitter_50, glove_twitter_200, humanStandards, randomBaseline]

# Create a bar chart showing the differences between the models
plt.bar(x, y)
plt.xlabel("Model Name")
plt.ylabel("Performance")
plt.title("Model Performance")
plt.savefig("model-performance.pdf")