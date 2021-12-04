import random

from gensim import downloader as api
from csv import reader

# this step is taking in the model and creating vectors to be used in the next steps
print("importing model")
model = api.load("word2vec-wikiDump-2017-300")  # load takes about 40 seconds - give it time
print("model imported")

vocabSize = 0
C = 0
V = 0

# section 1 of task 1
with open('synonyms.csv', 'r') as synonyms:     # opening the synonyms file
    csvReader = reader(synonyms)        # csv reader to be used inn next steps
    lineNum = 0
    vocab = model.index_to_key      # list of the words in the pretrained model - used to know if the words we want to find are in the model or not
    vocabSize = len(vocab)
    for line in csvReader:      # read the file one line at a time
        if lineNum != 0:        # ignore the first line that is basically headers
            questionWord = line[0]      # this line and next 3, extract data to use from line
            rightAnswer = line[1]
            fourOptions = [line[2], line[3], line[4], line[5]]

            questionWordFound = None       # these two lines check if words are in model
            optionsFound = 0

            options = []        # to hold the multiple choices

            if questionWord in vocab:       # these lines assign value depending on if questionWord in model
                questionWordFound = True
            else:
                questionWordFound = False

            for option in fourOptions:      # these lines assign word options to options[] if found in model
                if option in vocab:
                    optionsFound += 1
                    options.append(option)

            # label guess if questionWord or none of the fourOptions were found in vocab
            if questionWordFound is False or len(options) == 0:
                # system performs random guess here
                guessedIndex = random.randint(0,3)      # generate a random number between 0-3
                guessWord = fourOptions[guessedIndex]   # use this index so choose an answer at random
                result = questionWord+ ", "+ rightAnswer+ ", "+ guessWord+ ", guess\n"
                file = open("word2vec-wikiDump-2017-300-details.csv", "a")    # output result to file
                file.write(result)
                file.close()

            # if we found the questionWord and at least one of the option words in model
            elif questionWordFound is True and len(options) > 0:
                scores = []     # to store the scores of each word
                for option in options:      # iterate through the option words from synonyms.csv
                    scores.append(model.similarity(questionWord, option))   # get the similarity score for each and store in scores[]
                winningIndex = scores.index(max(scores))    # find the index of the most similar word
                # label correct if questionWord and at least one of the fourOptions are in vocab AND guessed right
                if options[winningIndex] == rightAnswer:    # check if the word we guessed is the same as the right answer from synonyms.csv
                    C += 1      # counting right answers
                    V += 1      # counting words in vocab
                    result = questionWord+ ", "+ rightAnswer+ ", "+ options[winningIndex]+ ", correct\n"
                    file = open("word2vec-wikiDump-2017-300-details.csv", "a")        # output result to file
                    file.write(result)
                    file.close()
                # label wrong if questionWord and at least one of the fourOptions are in vocab AND guessed wrong
                elif options[winningIndex] != rightAnswer:  # check if the word we guessed is NOT the same as the right answer from synonyms.csv
                    V += 1      # counting words in vocab
                    result = questionWord+ ", "+ rightAnswer+ ", "+ options[winningIndex]+ ", wrong\n"
                    file = open("word2vec-wikiDump-2017-300-details.csv", "a")        # output result to file
                    file.write(result)
                    file.close()
        lineNum += 1

# section 2 of task 1
a = "\nword2vec-wikiDump-2017-300"      # the name of the model being used
comma = ", "
b = str(vocabSize)      # number of words in model
c = str(C)              # number of correct answers
d = str(V)              # size of vocabulary
e = str(C/V)            # accuracy of our system
file = open("analysis_1.csv", "a")        # following lines output to file
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