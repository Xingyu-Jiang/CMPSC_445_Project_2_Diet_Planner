import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(filepath):
    """Load the food dataset from a CSV file."""
    return pd.read_csv(filepath)


def get_tfidf(data):
    """Generate a TF-IDF matrix for the descriptions."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['Description'])
    return vectorizer, tfidf_matrix


def search_food(data, search_query, vectorizer, tfidf_matrix):
    """Search for food items matching the search query using TF-IDF vectorization."""
    query_tfidf = vectorizer.transform([search_query])
    cosine_similarities = (tfidf_matrix * query_tfidf.T).toarray().flatten()
    relevant_indices = cosine_similarities.argsort()[-10:][::-1]
    return data.iloc[relevant_indices]


def analyze_nutrition(data, food_id, targets):
    """Analyze the nutritional content and provide recommendations."""
    food_nutrition = data.loc[food_id]
    analysis = {}
    recommendations = []
    for nutrient, value in targets.items():
        actual_value = food_nutrition.get(nutrient, 0)
        analysis[nutrient] = actual_value
    return analysis


def display_options(data, search_query, vectorizer, tfidf_matrix):
    """Display food options based on the search query using TF-IDF."""
    matches = search_food(data, search_query, vectorizer, tfidf_matrix)
    if not matches.empty:
        sample_size = min(10, len(matches))
        options = matches['Description'].sample(sample_size).reset_index(drop=True)
        print(f"\nFood options containing '{search_query}':")
        for index, option in options.items():
            print(f"{index + 1}. {option}")
        while True:
            choice = input("Enter the number of your choice (or 'R' to refresh, 'Q' to quit): ")
            if choice.isdigit() and 0 < int(choice) <= sample_size:
                selected_description = options[int(choice) - 1]
                selected_food = matches[matches['Description'] == selected_description].iloc[0]
                return selected_food
            elif choice.lower() == 'r':
                return display_options(data, search_query, vectorizer, tfidf_matrix)
            elif choice.lower() == 'q':
                print("Exiting selection.")
                return None
            else:
                print("Invalid input, please try again.")
    else:
        print("No matching food found.")
        return None


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
        search_query = input("\nEnter a keyword to search for your diet food plan (or type 'exit' to quit): ")
        if search_query.lower() == 'exit':
            print("Exiting the program.")
            break
        selected_food = display_options(data, search_query, vectorizer, tfidf_matrix)
        if selected_food is not None:
            nutrition_info = analyze_nutrition(data, selected_food.name, nutritional_targets)
            for nutrient, value in nutrition_info.items():
                cumulative_nutrition[nutrient] += value
            print("\nCumulative Nutritional Intake:")
            for nutrient, value in cumulative_nutrition.items():
                print(f"{nutrient}: {value} grams (Target: {nutritional_targets[nutrient]} grams)")
                if value < nutritional_targets[nutrient]:
                    print(f"Consider increasing your intake of {nutrient.split('.')[-1]}\n")
                elif value > nutritional_targets[nutrient]:
                    print(f"Consider decreasing your intake of {nutrient.split('.')[-1]}\n")


if __name__ == "__main__":
    main()
