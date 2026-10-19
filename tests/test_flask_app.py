import unittest
from flask_app.app import app, model, vectorizer

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Accept titles that include 'Sentiment Analysis' (may include suffixes)
        self.assertIn(b'Sentiment Analysis', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data=dict(text="I love this!"))
        if model is None or vectorizer is None:
            self.assertEqual(response.status_code, 503)
            self.assertIn(b"Prediction service is not configured", response.data)
        else:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(b"Positive" in response.data or b"Negative" in response.data)

if __name__ == '__main__':
    unittest.main()
