import pandas as pd

import suspicious_chat_v2

dataset = pd.read_csv("labeled_data.csv")
ml = suspicious_chat_v2.SuspiciousDetection(dataset=dataset)

sentence = "You are the good girl."
ml.predict_sentence(sentence)

