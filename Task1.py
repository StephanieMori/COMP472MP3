import random

from gensim import downloader as api
from csv import reader

print("importing model")
model = api.load("word2vec-google-news-300")  # load takes about 30 seconds - give it time
print("model imported")

vocabSize = 0
C = 0
V = 0

# section 1 of task 1
with open('synonyms.csv', 'r') as synonyms:
    csvReader = reader(synonyms)
    lineNum = 0
    vocab = model.index_to_key
    vocabSize = len(vocab)
    for line in csvReader:
        if lineNum != 0:
            questionWord = line[0]
            rightAnswer = line[1]
            fourOptions = [line[2], line[3], line[4], line[5]]

            questionWordFound = None
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
                # random guess here
                guessedIndex = random.randint(0,3)
                guessWord = fourOptions[guessedIndex]
                result = questionWord+ ", "+ rightAnswer+ ", "+ guessWord+ ", guess\n"
                file = open("word2vec-google-news-300-details.csv", "a")
                file.write(result)
                file.close()

            elif questionWordFound is True and len(options) > 0:
                scores = []
                for option in options:
                    scores.append(model.similarity(questionWord, option))
                winningIndex = scores.index(max(scores))
                # label correct if questionWord and at least one of the fourOptions are in vocab AND guessed right
                if options[winningIndex] == rightAnswer:
                    C += 1
                    V += 1
                    result = questionWord+ ", "+ rightAnswer+ ", "+ options[winningIndex]+ ", correct\n"
                    file = open("word2vec-google-news-300-details.csv", "a")
                    file.write(result)
                    file.close()
                # label wrong if questionWord and at least one of the fourOptions are in vocab AND guessed wrong
                elif options[winningIndex] != rightAnswer:
                    V += 1
                    result = questionWord+ ", "+ rightAnswer+ ", "+ options[winningIndex]+ ", wrong\n"
                    file = open("word2vec-google-news-300-details.csv", "a")
                    file.write(result)
                    file.close()
        lineNum += 1

# section 2 of task 1
a = "word2vec-google-news-300"
comma = ", "
b = str(vocabSize)
c = str(C)
d = str(V)
e = str(C/V)
file = open("analysis.csv", "a")
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