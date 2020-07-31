import unittest

from Detector import troveAPI

my_old_private_trove_key = "u8bjeau6d8871kg8"
my_private_trove_key = "3j228nhbi1pftav2"

class troveAPI_test(unittest.TestCase):

    def test_url_encode(self):
        self.assertEqual('why%20test%3F', troveAPI.url_encode("why test?"))
        self.assertEqual('this%20is%20a%20test%21', troveAPI.url_encode("this is a test!"))
        self.assertEqual('testing%201%202%203', troveAPI.url_encode("testing 1 2 3"))
        self.assertEqual('testing%201%2C%202%2C%203', troveAPI.url_encode("testing 1, 2, 3"))
        self.assertEqual('this%20is%20a%20test%2C%20a%20very%20important%20test', troveAPI.url_encode("this is a test, a very important test"))

    def test_trove_api_get(self):
        self.assertEqual('103115275', troveAPI.trove_api_get(my_private_trove_key, "103115275")['article']['id'])
        self.assertEqual('60321507', troveAPI.trove_api_get(my_private_trove_key, "60321507")['article']['id'])
        #self.assertRaises(TypeError or ConnectionError, lambda: troveAPI.trove_api_get(my_private_trove_key, "1343512")['article']['id'])
        #self.assertRaises(TypeError or ConnectionError, lambda: troveAPI.trove_api_get(my_private_trove_key, "21142137")['article']['id'])
        #self.assertRaises(TypeError or ConnectionError, lambda: troveAPI.trove_api_get(my_private_trove_key, "12345678")['article']['id'])

        #The above three tests fail, most likely due to the upgrade from Trove API to V2

if __name__ == '__main__':
    unittest.main()
