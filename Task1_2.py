import random

import pandas as pd
from gensim import downloader as api
from csv import reader

# original model name: word2vec-google-news-300
from matplotlib import pyplot as plt


def bagOfWords(corpora):
    # this step is taking in the model and creating vectors to be used in the next steps

    model = api.load(corpora)  # load takes about 40 seconds - give it time
    print("model imported")

    vocabSize = 0
    C = 0
    V = 0

    # section 1 of task 1
    print("reading synonym file...")
    with open('synonyms.csv', 'r') as synonyms:
        csvReader = reader(synonyms)  # csv reader to be used in next steps
        lineNum = 0
        vocab = model.index_to_key  # list of the words in the pretrained model
        vocabSize = len(vocab)

        print("Guessing answers...")
        for line in csvReader:  # read the file one line at a time
            if lineNum != 0:  # ignore the first line that is basically headers
                questionWord = line[0]
                rightAnswer = line[1]
                fourOptions = [line[2], line[3], line[4], line[5]]

                questionWordFound = None  # these two lines check if words are in model
                optionsFound = 0
                options = []

                if questionWord in vocab:
                    questionWordFound = True
                else:
                    questionWordFound = False

                for option in fourOptions:
                    if option in vocab:
                        optionsFound += 1
                        options.append(option)

                # label guess if questionWord or none of the fourOptions were found in vocab
                if questionWordFound is False or len(options) == 0:
                    guessedIndex = random.randint(0, 3)  # generate a random number between 0-3
                    guessWord = fourOptions[guessedIndex]  # use index to choose answer at random
                    result = questionWord + ", " + rightAnswer + ", " + guessWord + ", guess\n"
                    file = open(f"{corpora}-details.csv", "a")
                    file.write(result)
                    file.close()

                # if we found the questionWord and at least one of the option words in model
                elif questionWordFound is True and len(options) > 0:
                    scores = []
                    for option in options:
                        scores.append(model.similarity(questionWord, option))  # get similarity score for each and store
                    winningIndex = scores.index(max(scores))
                    # label correct if questionWord and at least one of the fourOptions are in vocab AND guessed right
                    if options[winningIndex] == rightAnswer:  # check if word guessed is right answer from synonyms.csv
                        C += 1  # counting right answers
                        V += 1  # counting words in vocab
                        result = questionWord + ", " + rightAnswer + ", " + options[winningIndex] + ", correct\n"
                        file = open(f"{corpora}-details.csv", "a")
                        file.write(result)
                        file.close()
                    # label wrong if questionWord and at least one of the fourOptions are in vocab AND guessed wrong
                    elif options[winningIndex] != rightAnswer:  # check if word guessed is NOT right answer
                        V += 1  # counting words in vocab
                        result = questionWord + ", " + rightAnswer + ", " + options[winningIndex] + ", wrong\n"
                        file = open(f"{corpora}-details.csv", "a")
                        file.write(result)
                        file.close()
            lineNum += 1

    # section 2 of task 1
    a = corpora  # the name of the model being used
    comma = ", "
    b = str(vocabSize)  # number of words in model
    c = str(C)  # number of correct answers
    d = str(V)  # size of vocabulary
    e = str(C / V) + "\n"  # accuracy of system
    file = open("analysis.csv", "a")  # following lines output to file
    file.write(a)
    file.write(comma)
    file.write(b)
    file.write(comma)
    file.write(c)
    file.write(comma)
    file.write(d)
    file.write(comma)
    file.write(e)
    file.close()

    performanceRatio = round((C / V) * 100, 2)
    return performanceRatio


if __name__ == '__main__':
    bagOfWords(corpora="word2vec-google-news-300")

    # First model: glove-wiki-gigaword-300 (C1-E1)
    # Second model: fasttext-wiki-news-subwords-300 (C2-E2)
    # Third model: glove-twitter-50 (C3-E3)
    # Fourth model: glove-twitter-200 (C4-E4)
    # note: C1 /= C2 and C3 = C4 and  E1 = E2 and E3 /= E4.

    print("\n importing models:")
    # Load the models and make them go threw task1 calculations
    glove_wiki_gigaword_300 = bagOfWords(corpora="glove-wiki-gigaword-300")
    fasttext_wiki_news_subwords_300 = bagOfWords(corpora="fasttext-wiki-news-subwords-300")
    glove_twitter_50 = bagOfWords(corpora="glove-twitter-200")
    glove_twitter_200 = bagOfWords(corpora="glove-twitter-25")

    # calculate accuracy for
    accuracy_rate = pd.read_csv('Crowdsourced_Gold_Standard_for_MP3.csv', index_col=3)

    human_standards = accuracy_rate["Sucess Rate"].tolist()
    specify_human_standards = {x for x in human_standards if x == x}
    iterate_human_standards = [x[:4] for x in specify_human_standards]
    human_standards_fl = [float(i) for i in iterate_human_standards]  # display as float for accuracy

    # create a random baseline
    random_baseline = human_standards_fl
    for i in range(len(random_baseline)):  # iterate throw random base-line
        random_baseline[i] = round(random.uniform(0, 100), 1)
    # calculate the values of
    humanStandards = round((sum(human_standards_fl) / len(human_standards_fl)),
                           2)  # Calculates the average for the Crowd Sourced Gold Standard
    randomBaseline = round((sum(random_baseline) / len(random_baseline)),
                           2)  # Calculates the average for the random baseline

    x = ["glove-wiki-gigaword-300", "fasttext-wiki-news-subwords-300", "glove-twitter-200", "glove-twitter-25",
         "Crowd Sourced Gold Standard", "Random Baseline"]
    y = [glove_wiki_gigaword_300, fasttext_wiki_news_subwords_300, glove_twitter_50, glove_twitter_200, humanStandards,
         randomBaseline]

    # Create a bar chart showing the differences between the models
    plt.bar(x, y, color=['yellow'], width=0.5)

    plt.xlabel("Model Name")
    plt.ylabel("Performance")
    plt.title("Comparisons of Model Performance")
    plt.show()
    plt.savefig("model-performance.pdf")

    print("\nsuccessful!")
