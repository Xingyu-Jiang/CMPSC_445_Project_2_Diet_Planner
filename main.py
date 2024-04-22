import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess text by removing punctuation, stopwords, and applying lemmatization."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Lowercase, remove punctuation, and split into words
    words = [word.lower() for word in nltk.word_tokenize(text) if word not in string.punctuation]

    # Remove stopwords and apply lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(lemmatized_words)

def load_data(filepath):
    """Load the food dataset from a CSV file and preprocess text."""
    data = pd.read_csv(filepath)
    data['Keywords'] = data['Description'].apply(preprocess_text)
    return data


def get_tfidf(data):
    """Generate a TF-IDF matrix for the preprocessed keywords."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['Keywords'])
    return vectorizer, tfidf_matrix


def search_food(data, search_query, vectorizer, tfidf_matrix):
    """Search for food items matching the search query using TF-IDF vectorization."""
    query_tfidf = vectorizer.transform([preprocess_text(search_query)])
    cosine_similarities = (tfidf_matrix * query_tfidf.T).toarray().flatten()
    relevant_indices = cosine_similarities.argsort()[-10:][::-1]
    return data.iloc[relevant_indices]


def analyze_nutrition(data, food_id, targets):
    """Analyze the nutritional content and provide recommendations."""
    food_nutrition = data.loc[food_id]
    analysis = {nutrient: food_nutrition.get(nutrient, 0) for nutrient in targets}
    return analysis


def get_user_choice(sample_size):
    """Prompt user for their choice and handle the response."""
    while True:
        choice = input("Enter the number of your choice, 'R' to refresh, 'N' for a new search or 'Q' to quit: ")
        if choice.isdigit() and 0 < int(choice) <= sample_size:
            return int(choice)
        elif choice.lower() == 'r':
            return 'refresh'
        elif choice.lower() == 'n':
            return 'new_search'
        # elif choice.lower() == 'c':
        #     return 'calculate'
        elif choice.lower() == 'q':
            return 'quit'
        else:
            print("Invalid input, please try again.")


def display_options(data, search_query, vectorizer, tfidf_matrix):
    """Display food options based on the search query using TF-IDF."""
    matches = search_food(data, search_query, vectorizer, tfidf_matrix)
    if not matches.empty:
        sample_size = min(10, len(matches))
        options = matches['Description'].sample(sample_size).reset_index(drop=True)
        print(f"\nFood options containing '{search_query}':")
        for index, option in options.items():
            print(f"{index + 1}. {option}")
        choice = get_user_choice(sample_size)
        if isinstance(choice, int):
            selected_description = options[choice - 1]
            return 'selected', matches[matches['Description'] == selected_description].iloc[0]
        elif choice == 'refresh':
            return 'refresh', display_options(data, search_query, vectorizer, tfidf_matrix)
        elif choice == 'new_search':
            return 'new_search', None
        # elif choice == 'calculate':
        #     return 'calculate', None
        elif choice == 'quit':
            print("Exiting selection.")
            return 'quit', None
    else:
        print("No matching food found.")
    return None, None


def print_top_terms(vectorizer, tfidf_matrix, top_n=10):
    """Print top n terms from the TF-IDF matrix."""
    feature_names = vectorizer.get_feature_names_out()
    for i, doc in enumerate(tfidf_matrix):
        print(f"Document {i + 1}:")
        sorted_indices = tfidf_matrix[i].tocoo().col.argsort()[::-1][:top_n]
        top_terms = [(feature_names[index], doc[0, index]) for index in sorted_indices]
        for term, score in top_terms:
            print(f"{term}: {score:.4f}")
        print("\n")


def main():
    filepath = 'CSV\\food.csv'
    data = load_data(filepath)
    vectorizer, tfidf_matrix = get_tfidf(data)
    # print(data.columns)
    # print_top_terms(vectorizer, tfidf_matrix, top_n=5)  # Print the top 5 terms for each document

    cumulative_nutrition = {
        'Data.Carbohydrate': 0,
        'Data.Fiber': 0,
        'Data.Protein': 0,
        'Data.Cholesterol': 0
    }
    nutritional_targets = {
        'Data.Carbohydrate': 130,
        'Data.Fiber': 25,
        'Data.Protein': 50,
        'Data.Cholesterol': 300
    }
    while True:
        search_query = input("Enter what would you like to eat here (type 'exit' to quit and 'calculate' for total "
                             "nutrient value): ")
        if search_query.lower() == 'exit':
            print("Exiting the program.")
            break
        elif search_query.lower() == 'calculate':
            print("\nCumulative Nutritional Intake:")
            for nutrient, value in cumulative_nutrition.items():
                print(f"{nutrient}: {value} grams (Target: {nutritional_targets[nutrient]} grams)")
                if value < nutritional_targets[nutrient]:
                    print(f"Consider increasing your intake of {nutrient.split('.')[-1]}\n")
                elif value > nutritional_targets[nutrient]:
                    print(f"Consider decreasing your intake of {nutrient.split('.')[-1]}\n")
        else:
            while True:
                action, selected_food = display_options(data, search_query, vectorizer, tfidf_matrix)
                if action == 'selected' and selected_food is not None:
                    nutrition_info = analyze_nutrition(data, selected_food.name, nutritional_targets)
                    for nutrient, value in nutrition_info.items():
                        cumulative_nutrition[nutrient] += value
                    break
                # elif action == 'calculate':
                #     print("\nCumulative Nutritional Intake:")
                #     for nutrient, value in cumulative_nutrition.items():
                #         print(f"{nutrient}: {value} grams (Target: {nutritional_targets[nutrient]} grams)")
                #         if value < nutritional_targets[nutrient]:
                #             print(f"Consider increasing your intake of {nutrient.split('.')[-1]}\n")
                #         elif value > nutritional_targets[nutrient]:
                #             print(f"Consider decreasing your intake of {nutrient.split('.')[-1]}\n")
                #     break
                elif action == 'new_search':
                    break  # Breaks inner loop, returns to search query input
                elif action == 'quit':
                    return  # Exits the program


if __name__ == "__main__":
    main()
