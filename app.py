from flask import Flask, request, render_template, redirect, url_for, flash, session, send_file

import tempfile
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pickle
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from flask_mail import Mail, Message

# Initialize Flask app and other configurations
app = Flask(__name__)

# SQLite database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Use SQLite and store in users.db
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '12345'
# app.secret_key = '12345'  # For flash messages
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your email account address'  # Your email address
app.config['MAIL_PASSWORD'] = 'generated app-key password'  # Your email password
mail = Mail(app)

# Initialize the SQLAlchemy object
db = SQLAlchemy(app)

# Initialize stemmer
stemmer = PorterStemmer()

clf = pickle.load(open('naive_bayes_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Define User model
class User(db.Model):
    __tablename__ = 'users'  # Explicitly set the table name
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="user")  # "admin" for admin users, "user" for regular users

    def __repr__(self):
        return f'<User {self.name}>'


class History(db.Model):
    __tablename__ = 'history'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationship to User
    user = db.relationship('User', backref=db.backref('histories', lazy=True))

    def __repr__(self):
        return f'<History {self.text}>'

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Message {self.id}>'

# Route for the contact form
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        # Create a new Message object
        new_message = Message(name=name, email=email, message=message)
        # Save to the database
        db.session.add(new_message)
        db.session.commit()
        # Flash message for success
        flash('Your message has been sent!', 'success')
        return redirect(url_for('contact'))
    return render_template('index.html')

# Create the database and tables if they don't exist
with app.app_context():
    db.create_all()  # This creates the 'users.db' file and 'User' table if it doesn't exist
    print("Database and tables created successfully!")

# for getting data from a dataset file.
stopwords_set = set(stopwords.words('english'))
emoji_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')

# Preprocessing function (same as used during training)
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Replace multi-word negation phrases
    text = re.sub(r'not\s+so\s+bad', 'not_so_bad_positive', text)
    # Explicitly handle simple phrases that are commonly positive or negative
    phrase_replacements = {
        "I am feeling tensed" : "negative_phase",
        "i like it": "positive_phrase",  # Replace "I like it" with a known positive tag
        "i love it": "positive_phrase",  # Replace "I love it" with a known positive tag
        "i don't like it": "negative_phrase",  # Replace "I don't like it" with a known negative tag
        "i hate it": "negative_phrase",  # Replace "I hate it" with a known negative tag
    }
    # Replace phrases in the text
    for phrase, replacement in phrase_replacements.items():
        text = text.replace(phrase, replacement)
    # Remove non-alphabetic characters (special characters and numbers)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords but keep negations
    stop_words = set(stopwords.words('english'))- {"not", "don't", "no"}
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Join the tokens back into a string
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        comment = request.form['text']
        if len(comment) > 50:
            if 'id' not in session:
                flash('You need to log in to analyze comments longer than 50 characters!', 'warning')
                return redirect(url_for('login'))  # Redirect to login page
        print(f"Input Comment: {comment}")
        # Preprocess the input text
        cleaned_text = preprocess_text(comment)
        print(f"Preprocessed Text: {cleaned_text}")
        # Explicit phrase handling
        if "positive_phrase" in cleaned_text:
            prediction = "Positive"
        elif "negative_phrase" in cleaned_text:
            prediction = "Negative"
        else:
            # Vectorize and predict
            text_vector = tfidf.transform([cleaned_text])
            model_prediction = clf.predict(text_vector)
            print(f"Model Prediction: {model_prediction}")
            prediction = "Positive" if model_prediction == 1 else "Negative"
        print(f"Final Prediction: {prediction}")
        return render_template('index.html', prediction=prediction, comment=comment)
    return render_template('index.html')

@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    if request.method == 'POST':
        comment = request.form['text']
        cleaned_comment = preprocess_text(comment)
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
        # Check if user is admin or normal user and redirect accordingly
        if user.role == 'admin':
            # Pagination parameters
            page = request.args.get('page', 1, type=int)
            per_page = 5  # Number of items per page
            # Paginate queries
            users_pagination = User.query.paginate(page=page, per_page=per_page, error_out=False)
            histories_pagination = History.query.paginate(page=page, per_page=per_page, error_out=False)
            messages_pagination = Message.query.paginate(page=page, per_page=per_page, error_out=False)
            return render_template('admin_dashboard.html',messages_pagination=messages_pagination, users_pagination=users_pagination, histories_pagination=histories_pagination,prediction=prediction, user=user, comment=comment)  # Redirect to admin dashboard
        else:
            # Get search query from URL
            search_query = request.args.get('search', '')
            page = request.args.get('page', 1, type=int)  # Get the current page from the query parameters
            per_page = 8  # Number of items per page
            # Default empty pagination object in case of no data
            paginated_histories = None
            if search_query:
                paginated_histories = History.query.filter(
                    History.user_id == user.id,
                    (History.text.ilike(f"%{search_query}%") | History.prediction.ilike(f"%{search_query}%"))
                ).order_by(History.id.desc()).paginate(page=page, per_page=per_page)
            else:
                paginated_histories = History.query.filter_by(user_id=user.id).order_by(History.id.desc()).paginate(
                    page=page, per_page=per_page)
            return render_template('user_dashboard.html', paginated_histories=paginated_histories, search_query=search_query, prediction=prediction, user=user, comment=comment)  # Redirect to user dashboard
    return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Query user by email
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            # Set user session
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_role'] = user.role  # Store role for use in templates
            flash('Login successful!', 'success')
            # Check if user is admin or normal user and redirect accordingly
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))  # Redirect to admin dashboard
            else:
                return redirect(url_for('user_dashboard'))  # Redirect to user dashboard
        else:
            flash('Invalid email or password!', 'danger')
            return redirect(url_for('login'))  # Stay on login page if credentials are invalid
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get data from form
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        # Check if the passwords match
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        # Check if user already exists
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists!', 'danger')
            return redirect(url_for('register'))
        # Hash the password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        # Check if the database has any users (admin or not)
        user_count = User.query.count()
        # If this is the first user, assign them as admin
        if user_count == 0:
            role = 'admin'  # First user is admin
        else:
            role = 'user'  # Other users are regular users
        # Create new user and add to the database
        new_user = User(name=name, email=email, password=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/user_dashboard', methods=['GET'])
def user_dashboard():
    if 'user_id' not in session:
        flash('You need to log in first!', 'warning')
        return redirect(url_for('login'))
    user_id = session['user_id']
    user = User.query.get(user_id)
    if user.role != 'user':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))
    # Get search query from URL
    search_query = request.args.get('search', '')
    page = request.args.get('page', 1, type=int)  # Get the current page from the query parameters
    per_page = 8  # Number of items per page
    # Default empty pagination object in case of no data
    paginated_histories = None
    if search_query:
        paginated_histories = History.query.filter(
            History.user_id == user.id,
            (History.text.ilike(f"%{search_query}%") | History.prediction.ilike(f"%{search_query}%"))
        ).order_by(History.id.desc()).paginate(page=page, per_page=per_page)
    else:
        paginated_histories = History.query.filter_by(user_id=user.id).order_by(History.id.desc()).paginate(page=page, per_page=per_page)
    return render_template('user_dashboard.html', user=user, paginated_histories=paginated_histories, search_query=search_query)

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_id' not in session:
        flash('You need to log in first!', 'warning')
        return redirect(url_for('login'))
    user_id = session['user_id']
    user = User.query.get(user_id)
    if user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = 5  # Number of items per page
    # Paginate queries
    users_pagination = User.query.paginate(page=page, per_page=per_page, error_out=False)
    histories_pagination = History.query.paginate(page=page, per_page=per_page, error_out=False)
    messages_pagination = Message.query.paginate(page=page, per_page=per_page, error_out=False)
    return render_template(
        'admin_dashboard.html',
        user=user,
        users_pagination=users_pagination,
        histories_pagination=histories_pagination,
        messages_pagination=messages_pagination,
    )

@app.route('/logout')
def logout():
    session.clear()  # Clear session data
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))  # Redirect to login page after logout

# Serializer for generating secure reset tokens
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Route for requesting password reset
# Define sender in config
app.config['MAIL_DEFAULT_SENDER'] = app.config['MAIL_USERNAME']  # Use Gmail username as sender

# In the reset_password route
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user:
            token = s.dumps(user.email, salt='password-reset-salt')
            reset_url = url_for('reset_with_token', token=token, _external=True)
            msg = Message('Password Reset Request', recipients=[email])
            msg.body = f'Click the link to reset your password: {reset_url}'
            # Use the default sender configured in app
            msg.sender = app.config['MAIL_USERNAME']  # Ensure sender is set
            try:
                mail.send(msg)
                flash('A password reset link has been sent to your email.', 'info')
                return redirect(url_for('login'))
            except Exception as e:
                flash(f'Error sending email: {str(e)}', 'danger')
                return redirect(url_for('reset_password'))
        else:
            flash('Email not found!', 'danger')
    return render_template('reset_password.html')

# Route for resetting password with token
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_with_token(token):
    try:
        # Verify the token
        email = s.loads(token, salt='password-reset-salt', max_age=3600)  # Token is valid for 1 hour
    except SignatureExpired:
        flash('The password reset link has expired. Please request a new one.', 'danger')
        return redirect(url_for('reset_password'))
    if request.method == 'POST':
        new_password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user:
            # Hash the new password and save it to the database
            hashed_password = generate_password_hash(new_password)
            user.password = hashed_password
            db.session.commit()
            flash('Your password has been updated!', 'success')
            return redirect(url_for('login'))
    return render_template('reset_with_token.html', token=token)

@app.route('/delete_history/<int:history_id>', methods=['POST'])
def delete_history(history_id):
    if 'user_id' not in session:
        flash('You need to log in first!', 'warning')
        return redirect(url_for('login'))
    # Query the history entry to delete
    history = History.query.get_or_404(history_id)
    # Check if the logged-in user owns this history record
    if history.user_id != session['user_id']:
        flash('You are not authorized to delete this history entry!', 'danger')
        return redirect(url_for('user_dashboard'))
    # Delete the history entry
    db.session.delete(history)
    db.session.commit()
    flash('History entry deleted successfully!', 'success')
    return redirect(url_for('user_dashboard'))

@app.route('/delete_user_history/<int:history_id>', methods=['POST'])
def delete_user_history(history_id):
    if 'user_id' not in session:
        flash('You need to log in first!', 'warning')
        return redirect(url_for('login'))
    # Query the history entry to delete
    history = History.query.get_or_404(history_id)
    # Check if the logged-in user owns this history record
    if history.user_id != session['user_id']:
        flash('You are not authorized to delete this history entry!', 'danger')
        return redirect(url_for('user_dashboard'))
    # Delete the history entry
    db.session.delete(history)
    db.session.commit()
    flash('History entry deleted successfully!', 'success')
    return redirect(url_for('user_dashboard'))

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'user_id' not in session:
        flash('You need to log in first!', 'warning')
        return redirect(url_for('login'))
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/delete_all_users_history/<int:history_id>', methods=['POST'])
def delete_all_users_history(history_id):
    # Ensure the user is logged in
    if 'user_id' not in session:
        flash('You need to log in first!', 'warning')
        return redirect(url_for('login'))
    # Get the logged-in user's details
    user_id = session['user_id']
    user = User.query.get(user_id)
    # Verify the logged-in user is an admin
    if user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))
    # Query the history record to delete
    history = History.query.get_or_404(history_id)
    # Delete the history entry
    db.session.delete(history)
    db.session.commit()
    flash('History entry deleted successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

# Implement admin login check (use session or token for real apps)
def is_admin_logged_in():
    # Example check for admin session or role
    return True  # Simplify for now

@app.route('/delete_message/<int:message_id>', methods=['POST'])
def delete_message(message_id):
    # Ensure the user is logged in
    if 'user_id' not in session:
        flash('You need to log in first!', 'warning')
        return redirect(url_for('login'))
    # Get the logged-in user's details
    user_id = session['user_id']
    user = User.query.get(user_id)
    # Verify the logged-in user is an admin
    if user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('admin_dashboard'))
    # Find the message by ID
    message = Message.query.get_or_404(message_id)
    # Delete the message
    db.session.delete(message)
    db.session.commit()
    flash('Message deleted successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/generate_report', methods=['GET'])
def generate_report():
    # Query the database for all history records
    users = User.query.all()
    histories = History.query.all()
    # Initialize counts
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    total_texts = len(histories)
    # Set to track unique users based on history
    total_users = len(users)
    # Loop through histories to count positive, negative, and neutral predictions
    for history in histories:
        if history.prediction.lower() == 'positive':
            positive_count += 1
        elif history.prediction.lower() == 'negative':
            negative_count += 1
        else:
            neutral_count += 1
    # # Get number of unique users
    # unique_users_count = len(unique_users)
    # Create a PDF using ReportLab
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter  # Default letter size (8.5 x 11 inches)
    # Add some basic text to the PDF
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 72, f"Sentiment Analysis Report")
    c.drawString(72, height - 100, f"")
    c.drawString(72, height - 128, f"Total Number of Texts: {total_texts}")
    c.drawString(72, height - 156, f"Total Unique Users: {total_users}")
    c.drawString(72, height - 184, f"Positive Predictions: {positive_count}")
    c.drawString(72, height - 202, f"Negative Predictions: {negative_count}")
    # Generate and save a bar graph
    categories = ['Positive', 'Negative', 'Neutral']
    counts = [positive_count, negative_count, neutral_count]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(categories, counts, color=['green', 'yellow', 'blue'])
    ax.set_title('Bar-Graph')
    # Save the bar graph as an image in memory
    bar_img_buffer = BytesIO()
    plt.savefig(bar_img_buffer, format='PNG')
    bar_img_buffer.seek(0)
    # Save the bar graph image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image:
        temp_image_path = temp_image.name
        with open(temp_image_path, 'wb') as f:
            f.write(bar_img_buffer.read())
        # Draw the bar graph on the PDF
        c.drawImage(temp_image_path, 72, height - 450, width=400, height=200)
    # Generate and save a pie chart
    sentiment_counts = [positive_count, negative_count, neutral_count]
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%', colors=['green', 'red', 'yellow'])
    ax.set_title('Pie-Chart')

    # Save the pie chart as an image in memory
    pie_img_buffer = BytesIO()
    plt.savefig(pie_img_buffer, format='PNG')
    pie_img_buffer.seek(0)

    # Save the pie chart image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image:
        temp_image_path = temp_image.name
        with open(temp_image_path, 'wb') as f:
            f.write(pie_img_buffer.read())

        # Draw the pie chart on the PDF
        c.drawImage(temp_image_path, 72, height - 700, width=400, height=200)

    # Save the PDF to the buffer
    c.save()

    # Get the value from the buffer
    buffer.seek(0)

    # Send the PDF as a downloadable file
    return send_file(buffer, as_attachment=True, download_name="sentiment_report.pdf", mimetype="application/pdf")

@app.route('/admin_sentiment')
def admin_sentiment():
    # Sentiment Analysis page
    user = User.query.get(session['user_id'])  # Example user data
    return render_template('admin_sentiment.html', user=user)

@app.route('/admin_profile/', methods=['GET'])
def admin_profile():
    user = User.query.get(session['user_id'])

    return render_template('admin_profile.html', user=user)

@app.route('/admin_history')
def admin_history():
    if 'user_id' not in session:
        flash('You need to log in first!', 'warning')
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)

    if user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))

    # Get all users from the database
    users = User.query.all()
    # All History page
    all_histories = History.query.all()  # Example history data
    return render_template('admin_history.html', all_histories=all_histories, user=user, users=users)

@app.route('/admin_manage_users')
def admin_manage_users():
    # Manage Users page
    users = User.query.all()  # Example user data
    return render_template('admin_manage_users.html', users=users)
if __name__ == "__main__":
    app.run(debug=True)
