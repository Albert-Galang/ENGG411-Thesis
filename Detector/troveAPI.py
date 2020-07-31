import json
import urllib
import urllib.request
from time import sleep
from urllib.error import HTTPError

from backoff import on_exception, expo
from ratelimit import limits, RateLimitException


@on_exception(expo, RateLimitException, max_time=60)
@limits(calls=100, period=60)
def call_api(url):
    """Requests response from URL. Complies with call limit.

    :param String url: valid encoded url request
    :return: response
    :rtype: object
    """
    attempts = 0
    while attempts < 10:
        try:
            response = urllib.request.urlopen(url)
            return response
        except HTTPError as e:
            print("Encountered Error: ", e)
            print("Attempting to reconnect...")
            sleep(6)
            attempts += 1

    response = urllib.request.urlopen(url)
    if response.getcode() != 200:
        raise Exception('API response: {}'.format(response.getcode()))
    return response


def url_encode(t):
    """Encodes input string to comply with URL requirements

    :param t: input string
    :return: url encoded string
    """
    return urllib.request.pathname2url(t)


def trove_api_request(trove_key, zone, category, search_terms, min_year, max_year, s, n):
    """Makes a search request to the Trove API with the provided
    search_terms and returns the selected page of results in JSON format

    :param trove_key:
    :param zone:
    :param category:
    :param search_terms:
    :param min_year:
    :param max_year:
    :param s:
    :param n:
    :return:
    """

    search_terms_url = url_encode(search_terms)

    request = "https://api.trove.nla.gov.au/v2/result?key=" + trove_key + \
              "&zone=" + zone + \
              "&q=" + search_terms_url + \
              "%20date%3A%5B" + str(min_year) + \
              "%20TO%20" + str(max_year) + \
              "%5D&encoding=json&s=" + str(s) + \
              "&n=" + str(n) + \
              "&reclevel=full&include=articletext&l-australian&l-category=" + category

    # print(request)
    response = call_api(request)
    response_content = response.read()
    response_json = json.loads(response_content)
    return response_json


def trove_api_get(trove_key, article_id):
    """Makes an article request to the Trove API with the provided article_id, returning the result in JSON format

    :param trove_key:
    :param article_id:
    :return:
    """

    request = "https://api.trove.nla.gov.au/v2/newspaper/" + str(
        article_id) + "?key=" + trove_key + "&include=articletext&encoding=json&reclevel=full&include=articletext"
    response = call_api(request)
    response_content = response.read()
    print(request)

    try:
        response_json = json.loads(response_content)
        return response_json
    except:
        raise Exception
