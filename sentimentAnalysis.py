import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD

# Load the dataset from a CSV file
df = pd.read_csv(r"C:\Users\chidu\OneDrive\Desktop\ml Projects\datasets\Tweets.csv", encoding='ISO-8859-1')

print(df.columns)

# Drop the 'id' column and reassign the DataFrame
df = df.drop(['textID', 'text'], axis=1)
df = df.dropna(subset=['selected_text', 'sentiment'])

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Include both unigrams and bigrams in the feature set
    stop_words='english',  # Remove common English stop words
    max_df=0.85,  # Ignore terms that appear in more than 85% of the documents
    min_df=2,  # Ignore terms that appear in fewer than 2 documents
    sublinear_tf=True  # Use sublinear term frequency scaling
)

# Transform the 'tweet' column into TF-IDF features
vec = vectorizer.fit_transform(df['selected_text'])
x = vec  # Store the transformed features

# Apply dimensionality reduction using Truncated SVD
svd = TruncatedSVD(n_components=100)  # Set the number of components to retain
x_reduced = svd.fit_transform(vec)  # Fit and transform the TF-IDF features

# Scale the features to the range [0, 1] using MaxAbsScaler
scaler = MaxAbsScaler()
fit_x = scaler.fit_transform(x)  # Scale the TF-IDF features
y = df['sentiment']  # Extract the target variable (categories)

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(fit_x, y, test_size=0.2, random_state=42)

# Initialize the Bernoulli Naive Bayes classifier
GB = BernoulliNB()
GB.fit(X_train, y_train)  # Fit the model to the training data

# Evaluate the model on the test set and print the accuracy score
print("Model Accuracy on Test Set:", GB.score(X_test, y_test))


def check_sentiment(user_input):
    """Collects user input and checks sentiment using the trained model."""
    # Transform the user input into TF-IDF features
    user_vec = vectorizer.transform([user_input])  # Transform to TF-IDF
    user_vec_scaled = scaler.transform(user_vec)  # Scale the features

    # Predict the sentiment category
    prediction = GB.predict(user_vec_scaled)
    return prediction[0]  # Return the predicted category


# Example of collecting input from the user
user_input = input("Enter a sentence to check its sentiment: ")
predicted_sentiment = check_sentiment(user_input)
print(f"The predicted sentiment category is: {predicted_sentiment}")
