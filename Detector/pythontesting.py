import json

from nltk import ngrams

from Detector.articleProcessing import *

my_private_trove_key = "3j228nhbi1pftav2"
data = troveAPI.trove_api_get(my_private_trove_key, 103115275)

# Pretty print the JSON data
print(json.dumps(data, indent=4))

# Removing markup https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python/24618186#24618186
htmltext = data['article']['articleText']
soup = BeautifulSoup(htmltext, features="html.parser")

for script in soup(["script", "style"]):
    script.extract()

clean_text = soup.get_text()
print(clean_text)

# Removing regular expressions (used to break up hyphenated words)
clean_text = re.sub(r'[^\w]', ' ', clean_text)

# Tokenization of JSON Data
articletext = word_tokenize(clean_text)
print('This has been tokenized: ', articletext)

# Removing stop words https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
stop_words = set(stopwords.words('english'))

filtered_articletext = [w for w in articletext if w.lower() not in stop_words and w not in string.punctuation]  # w.lower() allows removal of stop words that are capitalized or contain capital letters
print('The stop words and punctuation have been removed: ', filtered_articletext)

# Stemming (porter vs snowball)
ps_stemmer = PorterStemmer()
print(ps_stemmer.stem('generously'))

sb_stemmer = SnowballStemmer("english")
print(sb_stemmer.stem('generously'))

# Lemmatization
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('michael'))

# Lemmatization takes a slightly longer time in comparison to stemming as it does morphological analysis, now we try it on multiple tokens.
# From research we already know we want to use lemmatization, but we try both just for comparisons sake.

# Stemming multiple tokens
stemmed_text = ' '.join(sb_stemmer.stem(token) for token in filtered_articletext)
print(stemmed_text)
stemmed_tokenized_text = word_tokenize(stemmed_text)

# Lemmatization of multiple tokens

lemmatized_text = ' '.join(lemmatizer.lemmatize(token) for token in filtered_articletext)
print(lemmatized_text)
lemmatized_tokenized_text = word_tokenize(lemmatized_text)

# Ngrams frequency from https://stackoverflow.com/questions/40669141/python-nltk-counting-word-and-phrase-frequency

all_counts = dict()
for size in 2, 3, 4, 5:
    all_counts[size] = FreqDist(ngrams(lemmatized_tokenized_text, size))

# print(all_counts[4].most_common(5))

# Word Frequency from https://www.nltk.org/book/ch01.html
fdist1 = FreqDist(filtered_articletext)
print(fdist1)
print(fdist1.most_common(50))


#print(process_text(my_private_trove_key, 100756186))



sample_text1 = 'There goes my best friend. I knew him since I was but a child, and we gallivanted through the years, pretending we were heroes. Now look at us, old and frail.'
sample_text2 = 'There is nothing worse than wanting to die, I know personally. It pains me to tell you this, but he is dead. The years have taken him. Freeze-dry that dude'
sample_text3 = 'Is this what you wanted? We loved you, we cherished you. You never thought about anybody else but yourself, you are selfish. You are a monster.'
sample_text4 = 'I hate this. I am panicking, my chest hurts. But I still love you. Do you still love me? Five years from now, would you die for me? Would you consume my pain?'

#sample_text1 = 'This is Romeo and Juliet'
#sample_text2 = 'this is another play'
#sample_text3 = 'and another'
#sample_text4 = 'and one more'

sample_docs = []

tokenized_data = pre_process(sample_text1)
sample_docs.append(tokenized_data)
tokenized_data = pre_process(sample_text2)
sample_docs.append(tokenized_data)
tokenized_data = pre_process(sample_text3)
sample_docs.append(tokenized_data)
tokenized_data = pre_process(sample_text4)
sample_docs.append(tokenized_data)
# print(sample_docs)

#tfidf_func(sample_docs)

#read_csv('perilAUS_events.csv')


date = '20/05/2019'
wordcount = 1390
word_frequency = "this, that"
tfidf = "hello: 0.2323"
article_dict = {"date": date,
                "word frequency": word_frequency,
                "wordcount": wordcount,
                "tf-idf": tfidf
                }

#print(article_dict)
x, y, z = compute_term_frequency(process_text(my_private_trove_key, 103115275))
print(y.most_common())
print(z)

# art_col = process_search(my_private_trove_key, 'newspaper', 'Article&bulkHarvest=true', 'cyclone', '1902', '1902', '*', 100, [], 0, 0, 0, 0)
art_col = process_search(my_private_trove_key, 'newspaper', 'Article&bulkHarvest=true', 'cyclone', '1905', '1905',
                             '*', 100)
print(art_col)
