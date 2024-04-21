import pandas as pd

def load_data(filepath):
    """Load the food dataset from a CSV file."""
    return pd.read_csv(filepath)

def search_food(data, search_query):
    """Search for food items matching the search query."""
    data['Description_lower'] = data['Description'].str.lower()
    search_query_lower = search_query.lower()
    return data[data['Description_lower'].str.contains(search_query_lower)]

def analyze_nutrition(data, food_id, targets):
    """Analyze the nutritional content and provide recommendations."""
    food_nutrition = data.loc[food_id]
    analysis = {}
    recommendations = []
    
    for nutrient, value in targets.items():
        actual_value = food_nutrition.get(nutrient, 0)
        analysis[nutrient] = actual_value
        if actual_value < value:
            recommendations.append(f"Increase {nutrient}")
        elif actual_value > value:
            recommendations.append(f"Decrease {nutrient}")
    
    return {
        "Analysis": analysis,
        "Recommendations": recommendations
    }

def display_options(data, search_query):
    """Display 10 food options based on the search query."""
    matches = search_food(data, search_query)
    
    if not matches.empty:
        print(f"\nFood options containing '{search_query}':")
        options = matches['Description'].sample(10).reset_index(drop=True)
        for index, option in options.items():
            print(f"{index + 1}. {option}")
        
        choice = input("Enter the number of your choice (or 'R' to refresh for different choice): ")
        
        if choice.isdigit() and int(choice) > 0 and int(choice) <= len(options):
            selected_food = matches[matches['Description'] == options[int(choice) - 1]].iloc[0]
            return selected_food.name
        elif choice.lower() == 'r':
            return display_options(data, search_query)
    else:
        print("No matching food found. Please try a different search.")
        return None

# Load and prepare data
data = load_data('food.csv')
# Get user input for keyword search
search_query = input("Enter a keyword to search for your deit food plan: ")

# Display options and handle user input
selected_food_id = display_options(data, search_query)

if selected_food_id is not None:
    nutritional_targets = {
        'Data.Carbohydrate': 300,
        'Data.Fiber': 30,
        'Data.Protein': 50,
        'Data.Cholesterol': 300
    }
    analysis_results = analyze_nutrition(data, selected_food_id, nutritional_targets)
    print("\nNutritional Analysis:")
    for nutrient in ['Data.Carbohydrate', 'Data.Fiber', 'Data.Protein', 'Data.Cholesterol']:
        value = analysis_results['Analysis'].get(nutrient, 0)
        print(f"{nutrient} (Target: {nutritional_targets[nutrient]} grams): {value}")
    print("\nDietary Recommendations:")
    for recommendation in analysis_results['Recommendations']:
        print(recommendation)
