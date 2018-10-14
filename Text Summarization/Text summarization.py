# Text Summarization using NLP

import bs4 as bs
import urllib.request
import re
import nltk
nltk.download('stopwords')
import heapq

#Get the data
source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_warming').read()

# Parsing the data/ creating BeautifulSoup object
soup = bs.BeautifulSoup(source,'lxml')

#Fetching data
text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text

#Preprocessing the text
text = re.sub(r"\[[0-9]*\]*", " ", text) #remove references ex. [1][2][22]
text = re.sub(r"s\+"," ", text)           #remove extra spaces
clean_text = text.lower()             #convert all words to lowercase             
clean_text = re.sub(r'\W',' ',clean_text)   #remove all non word characters 
clean_text = re.sub(r'\d',' ',clean_text)  #remove all digits
clean_text = re.sub(r'\s+',' ',clean_text)           #remove extra spaces

# Tokenize sentences
sentences = nltk.sent_tokenize(text) #Can't pass clean_text as clean_text doesn't contain
#any fullstops/punctuation marks to form sentences. 


# Stopword list
stop_words = nltk.corpus.stopwords.words('english')

#Create dictionary
word2count = {}

for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1

#Create histogram
for key in word2count.keys():
    word2count[key] = word2count[key] / max(word2count.values())
    
#Calculatng sentence scores
sent2score = {}

for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' ')) < 25:
            # To avoid sideeffects of long sentences
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]

# Gettings best 5 lines             
best_sentences = heapq.nlargest(5, sent2score, key=sent2score.get)

print('---------------------------------------------------------')
for sentence in best_sentences:
    print(sentence)
