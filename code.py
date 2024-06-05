import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import tarfile
import requests
from io import BytesIO

# Step 1: Download and Extract the Dataset
url = 'https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2'
response = requests.get(url)
file = tarfile.open(fileobj=BytesIO(response.content), mode="r:bz2")
file.extractall()

# Step 2: Load the Data
def load_data(folder):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), 'r', errors='ignore') as file:
                data.append(file.read())
    return data

spam_emails = load_data('spam')
ham_emails = load_data('easy_ham')

# Create a DataFrame
emails = pd.DataFrame({
    'email': spam_emails + ham_emails,
    'label': [1]*len(spam_emails) + [0]*len(ham_emails)
})

# Step 3: Preprocess the Data
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(emails['email'], emails['label'], test_size=0.3, random_state=42)

# Convert text data into numerical data using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Step 4: Train the Model
# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Step 5: Evaluate the Model
# Predict on the test set
y_pred = clf.predict(X_test_counts)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
