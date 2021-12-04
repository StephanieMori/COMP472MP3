# COMP472MP3

current things about the code
- Make sure to downgrade to python 3.8 because the gensim documentation asked for it, and the assignment also asked for it.
- Gensim library was giving me issues when I had downloaded it into the location where my project is --> so if you have it downloaded in your project directory and it does not work then download it globally. What I mean is to download it to your desktop in the commandline .


- Also, output files are open in append mode, this means that any time you run the code the outputs will be added to the file after what is already there. This is not ideal since it leads to big files with duplicates. I usually delete the filees that exist before running it so that it is recreated from scratch - cleaner and better for actual outputs to be submitted.
- "model = api.load("word2vec-google-news-300")  # load takes about 40 seconds - give it time
" is a complex step that takes about 30-45 seconds to run depending what else is running on your computer, give it time before thinking it is having errors (lol)