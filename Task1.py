from gensim import downloader as api

model = api.load("word2vec-google-news-300")    # load takes about 30 seconds - give it time






pairs = [
    ('car', 'minivan'),   # a minivan is a kind of car
    ('car', 'bicycle'),   # still a wheeled vehicle
    ('car', 'airplane'),  # ok, no wheels, but still a vehicle
    ('car', 'cereal'),    # ... and so on
    ('car', 'communism'),
]
for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, model.similarity(w1, w2)))

# similarity comparison also takes about 1 minute - give it time
#print(model.most_similar(positive=['car', 'minivan'], topn=5))

# print(model.most_similar("a"))
# print("line6")
# print(api.info('text8'))
