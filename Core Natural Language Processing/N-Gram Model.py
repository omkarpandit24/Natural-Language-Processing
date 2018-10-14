import random
import nltk

text = """Global warming or climate change has become a worldwide concern. It is gradually developing into an unprecedented environmental crisis evident in melting glaciers, changing weather patterns, rising sea levels, floods, cyclones and droughts. Global warming implies an increase in the average temperature of the Earth due to entrapment of greenhouse gases in the earth’s atmosphere."""

# Order of the grams
n = 3

# Our N-Grams
ngrams = {}

#Character grams
# Creating the model
for i in range(len(text)-n):
    gram = text[i:i+n]
    if gram not in ngrams.keys():
        ngrams[gram] = []
    ngrams[gram].append(text[i+n])
     
# Testing our N-Gram Model    
currentGram = text[0:n]
result = currentGram
for i in range(100):
    if currentGram not in ngrams.keys():
        break
    possibilities = ngrams[currentGram]
    nextItem = possibilities[random.randrange(len(possibilities))]
    result += nextItem
    currentGram = result[len(result)-n:len(result)]
  
print(result)

#Word grams
# Building the model
words = nltk.word_tokenize(text)
for i in range(len(words)-n):
    gram = ' '.join(words[i:i+n])
    if gram not in ngrams.keys():
        ngrams[gram] = []
    ngrams[gram].append(words[i+n])
    
# Testing the model
currentGram = ' '.join(words[0:n])
result = currentGram
for i in range(30):
    if currentGram not in ngrams.keys():
        break
    possibilities = ngrams[currentGram]
    nextItem = possibilities[random.randrange(len(possibilities))]
    result += ' '+nextItem
    rWords = nltk.word_tokenize(result)
    currentGram = ' '.join(rWords[len(rWords)-n:len(rWords)])

print(result)
































