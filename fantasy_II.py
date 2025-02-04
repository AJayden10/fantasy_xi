import pandas as pd
import requests
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# Fetch player data from Fantasy Premier League API
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(url)
data = response.json()

# Extract relevant player data (Player, Cost, Points, Position)
players_data = []
for player in data['elements']:
    players_data.append({
        'Player': player['web_name'],
        'Position': data['element_types'][player['element_type'] - 1]['singular_name'],
        'Cost': player['now_cost'] / 10,  # Cost is in tenths, so divide by 10 for easier reading
        'Points': player['total_points']
    })

# Create DataFrame with player data
df = pd.DataFrame(players_data)

# Define budget constraint
budget = 100  # Fantasy football budget (for Premier League)

# Define positions needed (this will vary depending on the league rules)
positions_needed = {
    'Forward': 3,  # Need 3 forwards
    'Midfielder': 4,  # Need 4 midfielders
    'Defender': 3,  # Need 3 defenders
    'Goalkeeper': 1  # Need 1 goalkeeper
}

# Initialize the Linear Programming Problem
model = LpProblem("Fantasy_Sports_Team_Optimization", LpMaximize)

# Define variables: Whether a player is selected (0 or 1)
player_vars = {
    player: LpVariable(player, cat="Binary")  # Binary variable (1 if selected, 0 if not)
    for player in df['Player']
}

# Objective: Maximize total points of the selected players
model += lpSum(player_vars[player] * df.loc[df['Player'] == player, 'Points'].values[0] for player in df['Player']), "Total_Points"

# Budget constraint: The total cost of selected players must not exceed the budget
model += lpSum(player_vars[player] * df.loc[df['Player'] == player, 'Cost'].values[0] for player in df['Player']) <= budget, "Budget_Constraint"

# Position constraints: Ensure the correct number of players in each position
for position, required in positions_needed.items():
    model += lpSum(player_vars[player] for player in df[df['Position'] == position]['Player']) == required, f"{position}_Constraint"

# Solve the optimization problem
model.solve()

# Output the selected players
selected_players = [player for player in df['Player'] if player_vars[player].varValue == 1]

# Display results
print("Optimal Fantasy Sports Team:")
for player in selected_players:
    print(f"Player: {player}")
    
print("Total Cost:", sum(df.loc[df['Player'] == player, 'Cost'].values[0] for player in selected_players))
print("Total Projected Points:", sum(df.loc[df['Player'] == player, 'Points'].values[0] for player in selected_players))
