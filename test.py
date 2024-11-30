from flask import Flask, request, render_template, redirect, url_for, flash, session, send_file

import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
from flask_sqlalchemy import SQLAlchemy

# Load the trained classifier and TF-IDF vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
# tfidf = pickle.load(open('sentiment_analysis.pkl', 'rb'))
# tfidf = pickle.load(open('vectprozer.pkl', 'rb'))

# for getting data from a dataset file.
stopwords_set = set(stopwords.words('english'))
emoji_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')


def preprocessing(selected_text):
    # Remove HTML tags
    selected_text = re.sub(r'<[^>]*>', '', selected_text)

    # Handle emojis (if needed)
    emojis = emoji_pattern.findall(selected_text)

    # Replace non-word characters with a space
    selected_text = re.sub(r'[\W+]', ' ', selected_text.lower()) + ' '.join(emojis).replace('-', '')

    # Tokenize the sentence
    words = word_tokenize(selected_text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords_set]

    # Handle Negation
    negation = False
    processed_words = []

    for word in lemmatized_words:
        if word in ['not', "don't", "can't", "never", "no", "nothing"]:
            negation = True
        else:
            if negation:
                processed_words.append(f"neg_{word}")  # Mark negated words
                negation = False
            else:
                processed_words.append(word)

    # Return the processed sentence
    return " ".join(processed_words)

#
# @app.route('/predict', methods=['POST', 'GET'])
# def predict():
#     if request.method == 'POST':
#         comment = request.form['text']
#         cleaned_comment = preprocessing(comment)
#         comment_vector = tfidf.transform([cleaned_comment])
#         prediction = clf.predict(comment_vector)
#         if prediction == 1:
#             prediction = "Positive"
#         else:
#             prediction = "Negative"
#
#         return render_template('index.html', prediction=prediction)
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        comment = request.form['text']
        print("length=", len(comment))
        # Check if the length of the comment is greater than 50 characters
        if len(comment) > 50:
            if 'id' not in session:
                flash('You need to log in to analyze comments longer than 50 characters!', 'warning')
                return redirect(url_for('login'))  # Redirect to login page
        # Preprocess the comment
        cleaned_comment = preprocessing(comment)
        comment_vector = tfidf.transform([cleaned_comment])
        prediction = clf.predict(comment_vector)

        # Convert prediction to readable format
        if prediction == 1:
            prediction = "Positive"
        else:
            prediction = "Negative"

        return render_template('index.html', prediction=prediction)


@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    if request.method == 'POST':
        comment = request.form['text']
        cleaned_comment = preprocessing(comment)
        comment_vector = tfidf.transform([cleaned_comment])
        prediction = clf.predict(comment_vector)
        if prediction == 1:
            prediction = "Positive"
        else:
            prediction = "Negative"

        # Check if the user is logged in
        if 'user_id' not in session:
            flash('You need to log in first!', 'warning')
            return redirect(url_for('login'))

        user_id = session['user_id']
        user = User.query.get(user_id)

        # Store the analysis in the History table
        new_history = History(text=comment, prediction=prediction, user_id=user_id)
        db.session.add(new_history)
        db.session.commit()

        # Pass the user object and history to the template
        return render_template('user_dashboard.html', prediction=prediction, user=user)
    return redirect(url_for('index'))
