## Goal of the Project

The project aims to create a program that allows users to search for food items based on textual queries and analyze their nutritional content. The application integrates data preprocessing, text analysis via TF-IDF vectorization, and user interaction to deliver a comprehensive tool for nutritional management.

## Significance of the Project

Diet Planner provides users with a tool that can help them make informed dietary choices. By combining text analysis with nutritional evaluation, this program makes it easy for users to understand the nutritional content of different foods and meals. This is helpful for people that are trying to be mindful of their diet, whether for personal health goals, weight loss, or specific dietary needs.

The program uses simple text analysis to match food items with a comprehensive nutritional database. This makes it useful for everyday decision-making about food choices without needing complex knowledge or technical skills.

## Installation and Instructions to Use

### Prerequisites:
- Python 3.10 or above
- Libraries: pandas, scikit-learn, nltk
  
### Installation:

- Clone the repository or download the ZIP file.
  Command:
  ```bash
  Git clone https://github.com/Xingyu-Jiang/CMPSC_445_Project_2_Diet_Planner
  ```
- Install required Python libraries using: `pip install pandas scikit-learn nltk`.
  Command:
  ```bash
  pip install pandas scikit-learn nltk
  ```
- Run the program
  Command：
  ```bash
  .\main.py
  ```

### Usage Instructions:

- Run the script using python in your desired IDE.
- Follow on-screen prompts to enter search queries or analyze nutritional content.

## Structure of the Code

<div align="center">
    <img src="https://github.com/Xingyu-Jiang/CMPSC_445_Project_2_Diet_Planner/assets/117769320/2e52e14d-9648-4a62-8822-34a36f254765" alt="Description of image">
</div>


The application is organized into several key functions, each handling specific aspects:

- **Data Loading and Preprocessing:** Functions to load and preprocess data.
- **TF-IDF Vectorization:** Function to convert preprocessed text into a TF-IDF matrix.
- **Search Functionality:** Allows users to search for food items based on queries.
- **Nutritional Analysis:** Analyzes and sums up the nutritional content of selected items.
- **User Interaction:** Handles user inputs and displays results.

## Functionalities and Test Results

### Functionalities:

- **Text-based search for food items:** Users can enter descriptions to find related food items.
- **Nutritional analysis:** Displays nutritional content and compares it with daily targets.
- **Interactive selection:** Users choose food items from search results to analyze further.

### Test Results:

- The text processing and TF-IDF vectorization were tested with various food descriptions to ensure accuracy.
- The search functionality was validated by checking the relevance of search results.
- Nutritional analysis was verified with the nutritional databases to ensure the calculations were correct.

## Discussion and Conclusions

The project effectively combines natural language processing with nutritional data analysis, creating a user-friendly tool for diet management. However, there are still certain challenges that have not been addressed yet. Notably, the system’s performance heavily depends on the accuracy of the input data, and the program’s reliability across more diverse food items.

---
