# Fantasy Football Player Points Predictor

## Overview
This project involves developing a machine learning model to predict fantasy football player point totals for the 2023 season based on previous year data. The model aims to provide accurate predictions to assist fantasy football enthusiasts in making informed decisions.

## Project Goals
- Utilize machine learning techniques to predict 2023 fantasy football player point totals.
- Compare predicted points against actual points to evaluate model performance.
- Achieve a high level of accuracy in predictions to enhance decision-making for fantasy football players.

## Technologies and Libraries
- **Python**: The primary programming language used.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For building and evaluating the machine learning model.
- **Matplotlib/Seaborn**: For data visualization.

## Files Included
- **FF Predictor Tool.py**: The main Python script containing the machine learning model and prediction logic.
- **Projection VS Actual.py**: The script used to compare projected points against actual points and calculate percentage differences.
- **Projected_vs_Actual_Fantasy_Points_2023.csv**: The dataset containing the projected fantasy points for 2023.
- **fantasy.csv**: Additional dataset used in the project.
- **fantasy_rz.csv**: Another dataset used in the project.

## Project Steps

### Step 1: Load and Prepare the Data
```python
import pandas as pd

# Load the projected fantasy points for 2023
projected_data = pd.read_csv("Projected_vs_Actual_Fantasy_Points_2023.csv")
```

### Step 2: Actual Fantasy Points Data
Include actual fantasy points for RBs, WRs, and TEs as provided in your scripts.

### Step 3: Combine and Compare Data
```python
# Combine all the actual fantasy points into one dictionary
actual_fantasy_points = {**actual_fantasy_points_rb, **actual_fantasy_points_wr, **actual_fantasy_points_te}

# Initialize a list to store percentage differences
percentage_differences = []

# Iterate through the projected data and compare with actual data
for index, row in projected_data.iterrows():
    player_name = row['Player']
    projected_points = row['Projected_Fantasy_Points']
    
    if player_name in actual_fantasy_points:
        actual_points = actual_fantasy_points[player_name]
        percentage_difference = ((actual_points - projected_points) / projected_points) * 100
        percentage_differences.append({
            'Player': player_name,
            'Projected_Fantasy_Points': projected_points,
            'Actual_Fantasy_Points': actual_points,
            'Percentage_Difference': percentage_difference
        })

# Convert the results into a DataFrame
results_df = pd.DataFrame(percentage_differences)

# Display the results
import ace_tools as tools; tools.display_dataframe_to_user(name="Fantasy Points Comparison", dataframe=results_df)
```

## Insights and Conclusions
The machine learning model achieved a 67.5% accuracy in predicting player point totals for 2023 using previous year's data, demonstrating the model's reliability and effectiveness. Key insights include:
- **Model Performance**: The model provides reasonably accurate predictions, assisting fantasy football players in their decision-making.
- **Data Utilization**: The project effectively combines historical data with machine learning techniques to generate useful predictions.

## How to Run the Project
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed (`pandas`, `scikit-learn`, `matplotlib`, `seaborn`).
3. Run the `FF Predictor Tool.py` script to generate predictions for the 2023 season.
4. Run the `Projection VS Actual.py` script to compare the predicted points against actual points and evaluate model performance.
5. Check the output for insights and conclusions based on the model's predictions.
