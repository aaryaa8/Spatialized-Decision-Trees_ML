import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import random

# Step 1: Install and import required libraries
# pip install spacy scikit-learn

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Step 2: Define functions for spatial category and quantitative subcategory extraction
def extract_spatial_category(doc):
    keywords = {"walking", "stairs", "ramps", "doors"}
    for token in doc:
        if token.lower_ in keywords:
            return token.lower_
    return None


def extract_quantitative_subcategory(doc, spatial_category):
    if spatial_category == "walking":
        keywords = {"easy", "moderate", "difficult"}
    elif spatial_category == "stairs":
        keywords = {"easy", "moderate", "difficult"}
    elif spatial_category == "ramps":
        keywords = {"easy", "moderate", "difficult"}
    elif spatial_category == "doors":
        keywords = {"easy", "moderate", "difficult"}
    else:
        return None

    for token in doc:
        if token.lower_ in keywords:
            return token.lower_
    return None


# Step 3: Define functions for processing text and extracting features
def process_text(text):
    return nlp(text).text


def extract_features(text):
    doc = nlp(text)

    # Extract spatial category
    spatial_category = extract_spatial_category(doc)

    # Extract quantitative subcategory
    quantitative_subcategory = extract_quantitative_subcategory(doc, spatial_category)

    return spatial_category, quantitative_subcategory


# Step 4: Generate 100 qualitative datasets
def generate_qualitative_datasets(num_samples=100):
    experiences = [
        "Collaborated on a high-profile project with a diverse team.",
        "Navigated a tight project deadline with successful outcomes.",
        "Experienced a flexible work schedule with remote work options.",
        # ... Add more experiences as needed
    ]

    random.shuffle(experiences)
    return experiences[:num_samples]


# Step 5: Save qualitative datasets to a CSV file
def save_to_csv(qualitative_datasets, filename="qualitative_datasets.csv"):
    df = pd.DataFrame({"Qualitative Dataset": qualitative_datasets})
    df.to_csv(filename, index=False)


# Step 6: Read CSV file and process each row
def process_csv_file(filename="qualitative_datasets.csv"):
    df = pd.read_csv(filename)

    results = []

    for index, row in df.iterrows():
        text = row["Qualitative Dataset"]
        spatial_category, quantitative_subcategory = extract_features(text)

        print(
            f"Row {index + 1} - Spatial Category: {spatial_category}, Quantitative Subcategory: {quantitative_subcategory}")

        results.append([text, spatial_category, quantitative_subcategory])

    return results


# Step 7: Train a machine learning model for spatial categories
def train_spatial_model(df):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df["Qualitative Dataset"], df["Spatial Category"],
                                                        test_size=0.2, random_state=42)

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Train a support vector machine (SVM) model
    svm_model = SVC(kernel="linear")
    svm_model.fit(X_train_tfidf, y_train)

    return svm_model, tfidf_vectorizer


# Step 8: Train a machine learning model for quantitative subcategories
def train_quantitative_model(df):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df["Qualitative Dataset"], df["Quantitative Subcategory"],
                                                        test_size=0.2, random_state=42)

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Train a support vector machine (SVM) model
    svm_model = SVC(kernel="linear")
    svm_model.fit(X_train_tfidf, y_train)

    return svm_model, tfidf_vectorizer


# Step 9: Run the script
# Generate qualitative datasets
qualitative_datasets = generate_qualitative_datasets(100)

# Save qualitative datasets to a CSV file
save_to_csv(qualitative_datasets)

# Process CSV file and extract features
results = process_csv_file("qualitative_datasets.csv")

# Create a DataFrame from the results
df_results = pd.DataFrame(results, columns=["Qualitative Dataset", "Spatial Category", "Quantitative Subcategory"])

# Train a machine learning model for spatial categories
svm_model_spatial, tfidf_vectorizer_spatial = train_spatial_model(df_results)

# Train a machine learning model for quantitative subcategories
svm_model_quantitative, tfidf_vectorizer_quantitative = train_quantitative_model(df_results)

# Provide a list of new qualitative experiences for prediction
new_experiences = [
    "Worked on a straightforward project with a relaxed timeline.",
    "Managed a complex project with tight deadlines and overtime.",
    "Collaborated with a diverse team on an innovative design.",
    # ... Add more new experiences as needed
]

# Predict spatial categories and quantitative subcategories for new experiences
for new_experience in new_experiences:
    spatial_category, quantitative_subcategory = extract_features(new_experience)
    print(f"New Experience: {new_experience}")
    print(f"Predicted Spatial Category: {spatial_category}")
    print(f"Predicted Quantitative Subcategory: {quantitative_subcategory}")

# Predict spatial categories and quantitative subcategories for new experiences and update the CSV file
predict_new_experiences(new_experiences, svm_model_spatial, svm_model_quantitative, tfidf_vectorizer_spatial,
                        "results.csv")
