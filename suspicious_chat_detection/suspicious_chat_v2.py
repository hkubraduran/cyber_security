import joblib
import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder


class SuspiciousDetection:

    def __init__(self, dataset=None):
        self.classifier_rf = None
        self.cv = None
        self.dataset = dataset

    def preprocess(self, data=None):
        dt_transformed = data[['class', 'tweet']]
        y = dt_transformed.iloc[:, :-1].values

        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
        y = np.array(ct.fit_transform(y))

        y_df = pd.DataFrame(y)
        y_hate = np.array(y_df[0])
        y_off = np.array(y_df[1])

        cleaned = self.clean_text(dt_transformed)
        self.cv = CountVectorizer(max_features=2000)
        x = self.cv.fit_transform(cleaned).toarray()

        return x, y_off

    def clean_text(self, dt_transformed):
        cleaned = []
        for i in range(0, 24783):
            review = re.sub('[^a-zA-Z]', ' ', dt_transformed['tweet'][i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            cleaned.append(review)
        return cleaned

    def train_model(self):
        X, y_offensive = self.preprocess(self.dataset)

        X_train, X_test, y_train, y_test = train_test_split(X, y_offensive, test_size=0.20, random_state=0)

        # Fitting Random Forest
        self.classifier_rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        self.classifier_rf.fit(X_train, y_train)
        y_pred_rf = self.classifier_rf.predict(X_test)

        # Fitting Naive Bayes
        # classifier_np = GaussianNB()
        # classifier_np.fit(X_train, y_train)
        # y_pred_np = classifier_np.predict(X_test)

        # Fitting Naive Bayes with Multinomial
        # classifier_np_m = MultinomialNB()
        # classifier_np_m.fit(X_train, y_train)
        # y_pred_np_m = classifier_np_m.predict(X_test)

        # Accuracy scores for each classifier
        # np_score = accuracy_score(y_test, y_pred_np)
        # np_score_m = accuracy_score(y_test, y_pred_np_m)
        rf_score = accuracy_score(y_test, y_pred_rf)

        print('Random Forest Accuracy: ', str(rf_score))
        # print('Naive Bayes Accuracy: ', str(np_score))
        # print('Naive Bayes Accuracy (Multinomial): ', str(np_score_m))

    def save_model(self, model_path="pretrained_model.joblib"):
        # Modeli ve vektörizeri kaydet
        joblib.dump((self.classifier_rf, self.cv), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path="pretrained_model.joblib"):
        # Kaydedilmiş modeli ve vektörizeri yükle
        try:
            self.classifier_rf, self.cv = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Pre-trained model not found at {model_path}. Train the model first.")

    def predict_sentence(self, word):
        # Convert input text to vector
        test_vectorized = self.cv.transform([word]).toarray()

        # Predictions for each classifier
        prediction_rf = self.classifier_rf.predict(test_vectorized)
        # prediction_np = classifier_np.predict(test_vectorized)
        # prediction_np_m = classifier_np_m.predict(test_vectorized)
        return prediction_rf

        # print('Random Forest Prediction:', 'Hate' if prediction_rf == 1 else 'Not Hate')
        # print('Naive Bayes Prediction:', 'Hate' if prediction_np == 1 else 'Not Hate')
        # print('Naive Bayes (Multinomial) Prediction:', 'Hate' if prediction_np_m == 1 else 'Not Hate')
