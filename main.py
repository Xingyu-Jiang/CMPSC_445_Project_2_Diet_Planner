import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')


def load_data(filepath):
    """ Load the recipe data from a CSV file. """
    return pd.read_csv(filepath)


def process_input(user_input):
    """ Process user input to extract relevant keywords. """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(user_input)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return filtered_words


def search_recipes(keywords, data):
    """ Search for recipes that match the keywords. """

    # Filter function to match keywords with multiple columns
    def match_keywords(row):
        text = f"{row['Recipe_name']} {row['Diet_type']} {row['Cuisine_type']}".lower()
        return any(keyword in text for keyword in keywords)

    matched_recipes = data[data.apply(match_keywords, axis=1)]
    return matched_recipes


def main():
    # Load the data
    filepath = 'CSV\\All_Diets.csv'  # Update the path to your CSV file
    data = load_data(filepath)

    while True:
        # User input
        user_input = input("Enter keywords for recipe search or type 'Exit now' to quit: ")
        if user_input.lower() == "exit now":
            break
        keywords = process_input(user_input)

        # Search recipes
        matched_recipes = search_recipes(keywords, data)

        # Display results
        if not matched_recipes.empty:
            print("Found Recipes:")
            print(matched_recipes[['Recipe_name', 'Diet_type', 'Cuisine_type', 'Protein(g)', 'Carbs(g)', 'Fat(g)']])
        else:
            print("No recipes found matching your criteria.")


if __name__ == '__main__':
    main()
