import csv
import string
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from nltk import FreqDist
from nltk.corpus import stopwords, words
from nltk.stem import *
from nltk.tokenize import *
from sklearn.feature_extraction.text import TfidfVectorizer

from Detector import troveAPI


def read_csv(filename):
    """Reads csv file, not inputting any with missing date information up to day i.e YYYY/MM/DD, hour is not recorded

    :param filename:
    :return peril_list: A list of dictionaries with peril and date in datetime format
    """
    reader = csv.DictReader(open(filename))
    peril_list = []

    for row in reader:
        if row['Start Year'] == 'NULL' or row['Start Year'] == '0' or \
                row['Start Month'] == 'NULL' or row['Start Month'] == '0' or \
                row['Start Day'] == 'NULL' or row['Start Day'] == '0':
            continue

        peril = row['Peril Type']
        date = row['Start Year'] + '-' + row['Start Month'] + '-' + row['Start Day']
        try:
            date = datetime.strptime(date, '%Y-%m-%d')
            peril_dict = {'peril': peril,
                          'date': date}
            peril_list.append(peril_dict)
        except ValueError as e:
            print(e)
            continue

    return peril_list


def pre_process(text):
    """Pre-process/clean text. Removing markup, reg. expressions/punctuation, stop words, then tokenizing and applying lemmatization.

    :param text:
    :return: Text without stopwords, punctuation, markup and all lemmatized.
    """

    # Remove Markup
    soup = BeautifulSoup(text, features="html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()

    # Remove reg. ex. / punctuation, this also removes the hyphen in hyphenated words i.e freeze-dry -> freeze dry
    text = re.sub(r'[^\w]', ' ', text)

    # Tokenize and transform into lower case
    text = word_tokenize(text)
    text = [w.lower() for w in text]

    # Remove stop words
    english_words = set(words.words())
    stop_words = set(stopwords.words('english'))
    newstopwords = ['tho', 'mr', 'tbe', '000']
    stop_words.update(newstopwords)
    filtered_text = [w for w in text if
                     w.lower() in english_words and w.lower() not in stop_words and w not in string.punctuation and len(
                         w) > 2]
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = ' '.join(lemmatizer.lemmatize(token) for token in filtered_text)
    lemmatized_tokenized_text = word_tokenize(lemmatized_text)

    return lemmatized_tokenized_text


def compute_term_frequency(tokenized_text):
    """Calculate term count, term frequency and term frequency versus term count.

    :param tokenized_text:
    :return: return frequency of words and frequency of words against amount of words
    """
    # Frequency distribution
    term_count = len(tokenized_text)
    i = 0
    frequency_distribution = FreqDist(tokenized_text)
    tf = []
    while i < len(frequency_distribution):
        # Where [i][1] is the frequency of the word
        tf.append((frequency_distribution.most_common()[i][0], frequency_distribution.most_common()[i][1] / term_count))
        i = i + 1
    return term_count, frequency_distribution, tf


def process_text(trove_key, article_id):
    """UNUSED REDUNDANT function that fetches articletext given article ID and processes it. Pre-process and process key and article, grabs article text then processes it.

    :param article_id:
    :param trove_key:
    :return: list of most frequent tokens
    """
    data = troveAPI.trove_api_get(trove_key, article_id)
    text = data['article']['articleText']
    processed_text = pre_process(text)
    return processed_text


def process_search(trove_key, zone, category, search_terms, min_year, max_year, s, n):
    """Function that makes search requests to Trove. This allows every result to be processed. We now use a while loop instead of a recursive loop, as there is no limitation on iterations.

    :param trove_key:
    :param zone:
    :param category:
    :param search_terms:
    :param min_year:
    :param max_year:
    :param s: The key that goes to the next page of search results
    :param n: Number of results listed per page

    :return:
    """
    data = troveAPI.trove_api_request(trove_key, zone, category, search_terms, min_year, max_year, s, n)
    total = data['response']['zone'][0]['records']['total']
    print(total, 'items found')
    skip_year = 0
    year_skipped = False
    skip_pages = 0
    old_pages = 0
    pages = 0
    count = 0
    article_collection = []
    next_exists = True

    # Keep looping as long as there is another page to be processed
    while next_exists:
        if skip_pages == 0:
            i = 0
            record_size = data['response']['zone'][0]['records']['n']
            record_size = int(record_size)
            s = data['response']['zone'][0]['records']['nextStart']

            # Construction of the dictionary to be appended to the list
            while i < record_size and skip_pages == 0:
                try:
                    article_id = data['response']['zone'][0]['records']['article'][i]['id']
                    processed_text = pre_process(data['response']['zone'][0]['records']['article'][i]['articleText'])
                    date = data['response']['zone'][0]['records']['article'][i]['date']
                    date = datetime.strptime(date, '%Y-%m-%d')
                    term_count, word_frequency, term_frequency = compute_term_frequency(processed_text)
                    article_dictionary = {"date": date,
                                          "id": article_id,
                                          "processed text": processed_text,
                                          "term count": term_count,
                                          "word frequency": word_frequency,
                                          "term frequency": term_frequency,
                                          }
                    article_collection.append(article_dictionary)
                    i += 1
                    count += 1
                except KeyError as e:
                    print('Could not find key', e)
                    print('Skipping this article...')
                    i += 1
                    count += 1
            print(count, 'articles processed')
        else:
            skip_pages -= 1

        pages += 1
        data = troveAPI.trove_api_request(trove_key, zone, category, search_terms, min_year, max_year, s, n)
        total = data['response']['zone'][0]['records']['total']
        total = int(total)

        if skip_year >= 15:
            print('This year will be skipped, too many non-results returned from API!')
            year_skipped = True
            next_exists = False

        # If we encounter an error resulting in an empty page of results during another restart
        if total == 0 and skip_pages != 0:
            print('Error encountered during restart, empty page of results returned... Restarting search from scratch')
            if pages > old_pages - 3:
                skip_year += 1
            skip_pages = old_pages
            pages = 0
            s = '*'
            data = troveAPI.trove_api_request(trove_key, zone, category, search_terms, min_year, max_year, s, n)
            next_exists = True
        # For the first time we encounter the error, or when it occurs on an unexplored page
        elif total == 0 and skip_pages == 0:
            print('Error encountered, empty page of results returned... Restarting search from scratch')
            if old_pages == pages:
                skip_year += 1
                print('Same restart happened:', skip_year, 'times')
            else:
                skip_year = 0
            old_pages = pages
            skip_pages = pages
            pages = 0
            s = '*'
            data = troveAPI.trove_api_request(trove_key, zone, category, search_terms, min_year, max_year, s, n)
            next_exists = True
        # There is no error encountered
        elif 'nextStart' in data['response']['zone'][0]['records']:
            if skip_pages != 0:
                print('Skipping page: ', pages, '...', sep='')
            s = data['response']['zone'][0]['records']['nextStart']
        else:
            next_exists = False

    return article_collection, year_skipped


def identity_tokenizer(text):
    """Necessary function for tfidf to work

    :param text:
    :return:
    """
    return text


def tfidf_func(data, total):
    """Inputting a list of tokenized texts, performs sklearns tf-idf function. Prints a list of tf-idf values in comparison to the first document
    Code extracted from https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/. Some minor adjustments made such that the TfidfVectorizer accepts tokenized inputs.

    :param total: total amount of articles being processed
    :param data
    :return:
    """
    # settings that you use for count vectorizer will go here
    tfidf_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    # just send in all your docs here
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(data)
    # print(tfidf_vectorizer_vectors.shape)
    # get the first vector out (for the first document)
    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

    # place tf-idf values in a pandas data frame
    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
                      columns=[0])

    # iterate over each article to store into data frame
    i = 1
    int_total = int(total)
    while i < int_total:
        df[i] = tfidf_vectorizer_vectors[i].T.todense()
        i += 1

    return df

