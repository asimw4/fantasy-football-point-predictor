import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the data
fantasy_stats = pd.read_csv(r"C:\Users\asimw\Downloads\fantasy.csv")
redzone_stats = pd.read_csv(r"C:\Users\asimw\Downloads\fantasy_rz.csv")

# Merge the dataframes on 'Player' and 'Team'
merged_data = pd.merge(fantasy_stats, redzone_stats, on=['Player', 'Team'], how='inner')

# Convert columns to numeric, handling errors
numeric_columns = [
    'Passing Yards_x', 'Passing Touchdowns_x', 'Interceptions', 
    'Rushing Yards', 'Rushing Touchdowns', 'Receiving Yards', 
    'Receiving Touchdowns', 'Red Zone Targets', 'PPR Scoring Points'
]

for col in numeric_columns:
    merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

# Function to train and predict for a specific position
def train_and_predict(position, features):
    # Filter data for the specific position
    position_data = merged_data[merged_data['Fantasy Position ;)'] == position]
    
    if position_data.empty:
        print(f"No data found for position: {position}")
        return pd.DataFrame()

    # Scale PPR Scoring Points to give it more weight
    position_data.loc[:, 'PPR Scoring Points'] = position_data['PPR Scoring Points'] * 2

    X = position_data[features]
    y = position_data['Fantasy Points']

    # Fill missing values with 0 for simplicity
    X = X.fillna(0)
    y = y.fillna(0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model with adjusted parameters
    rf = RandomForestRegressor(n_estimators=200, max_depth=30, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate the model
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    print(f"\n{position} - Training MAE: {mean_absolute_error(y_train, y_pred_train)}")
    print(f"{position} - Testing MAE: {mean_absolute_error(y_test, y_pred_test)}")

    # Feature importances
    feature_importances = rf.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print(f"\n{position} - Feature Importances:")
    print(importance_df)

    # Predict the 2023 fantasy points
    position_data['Predicted Fantasy Points'] = rf.predict(X)
    
    return position_data[['Player', 'Predicted Fantasy Points']]

# Define feature sets for each position, including past PPR points and new features
rb_features = ['Rushing Yards', 'Rushing Touchdowns', 'Receiving Yards', 'Receiving Touchdowns', 'Games', 'PPR Scoring Points']
wr_features = ['Receiving Yards', 'Receiving Touchdowns', 'Red Zone Targets', 'Games', 'Pass Targets_x', 'PPR Scoring Points']
te_features = ['Receiving Yards', 'Receiving Touchdowns', 'Red Zone Targets', 'Games', 'Pass Targets_x', 'PPR Scoring Points']

# Train and predict for each position, excluding QBs
rb_predictions = train_and_predict('RB', rb_features)
wr_predictions = train_and_predict('WR', wr_features)
te_predictions = train_and_predict('TE', te_features)

# Combine all predictions
all_predictions = pd.concat([rb_predictions, wr_predictions, te_predictions])

# Save the projected data to a new CSV file in the Downloads directory, only including player name and predicted points
output_path = r"C:\Users\asimw\Downloads\Projected_Fantasy_Points_2023.csv"
all_predictions[['Player', 'Predicted Fantasy Points']].to_csv(output_path, index=False)

# Display the projected data
print("\nProjected Fantasy Points for 2023:")
print(all_predictions[['Player', 'Predicted Fantasy Points']].head())


