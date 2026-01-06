# Cricket Fielding Performance Analysis - IPL Match
# Advanced Level Data Analysis Project
# Based on IPL2367: Delhi Capitals Fielding Analysis

"""
PROJECT OVERVIEW:
This notebook analyzes fielding performance of Delhi Capitals players from IPL match.
We'll collect data, calculate performance scores using the official formula, and visualize insights.

RESEARCH QUESTION:
"How do different fielding positions and actions contribute to overall team defensive 
performance in IPL T20 matches, and which players demonstrate the highest fielding efficiency?"
"""

# ============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================================
# Explanation: We import libraries that help us work with data and create visualizations

import pandas as pd  # For handling data tables (like Excel but more powerful)
import numpy as np   # For mathematical calculations
import matplotlib.pyplot as plt  # For creating charts and graphs
import seaborn as sns  # For making beautiful statistical visualizations

# Configure visualization settings for better appearance
sns.set_style("whitegrid")  # Adds grid lines to charts
plt.rcParams['figure.figsize'] = (12, 6)  # Default chart size (width, height)

print("✓ Libraries imported successfully!")
print("Ready to analyze IPL fielding data...\n")

# ============================================================================
# STEP 2: CREATE DATASET BASED ON IPL SAMPLE
# ============================================================================
# Explanation: We create a realistic dataset for an IPL T20 match
# This represents ball-by-ball fielding data for Delhi Capitals

# Initialize data dictionary - each key becomes a column in our table
fielding_data = {
    'Match_No': [],
    'Innings': [],
    'Team': [],
    'Player_Name': [],
    'Ball_Count': [],
    'Position': [],
    'Short_Description': [],
    'Pick': [],
    'Throw': [],
    'Runs': [],
    'Over_Count': [],
    'Venue': []
}

# Sample data for 3 selected players across 20 overs (120 balls)
# Player 1: Rilee Russouw (Batsman who fields in inner circle)
# Player 2: Axar Patel (All-rounder, excellent fielder)
# Player 3: Kuldeep Yadav (Bowler, fields in various positions)

# Rilee Russouw - 40 fielding events
russouw_data = {
    'balls': 40,
    'positions': ['Short Mid Wicket', 'Mid On', 'Square Leg', 'Mid Wicket'] * 10,
    'picks': ['Y', 'Y', 'n', 'Y', 'Y', 'Y', 'n', 'Y', 'Y', 'Y',  # 10
              'Y', 'Y', 'Y', 'Y', 'n', 'Y', 'Y', 'Y', 'Y', 'Y',  # 10
              'Y', 'Y', 'Y', 'Y', 'Y', 'n', 'Y', 'Y', 'C', 'Y',  # 10
              'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'C'], # 10
    'throws': ['Y', 'N', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y',
               'Y', 'N', 'Y', 'N', 'N', 'Y', 'N', 'Y', 'Y', 'N',
               'Y', 'N', 'Y', 'Y', 'N', 'N', 'Y', 'N', '', 'Y',
               'Y', 'N', 'DH', 'Y', 'N', 'Y', 'N', 'Y', 'Y', ''],
    'runs': [1, 0, -1, 1, 0, 2, -1, 1, 0, 1,
             1, 0, 1, 0, -1, 1, 0, 2, 1, 0,
             1, 0, 1, 1, 0, -1, 1, 0, 3, 1,
             2, 0, 4, 1, 0, 1, 0, 1, 1, 3],
    'descriptions': ['Quick pickup', 'Stopped ball', 'Fumbled', 'Clean field', 'Routine stop'] * 8
}

# Axar Patel - 40 fielding events
axar_data = {
    'balls': 40,
    'positions': ['Point', 'Cover', 'Point', 'Cover Point'] * 10,
    'picks': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'C', 'Y',
              'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'C', 'Y', 'Y',
              'Y', 'Y', 'Y', 'Y', 'n', 'Y', 'Y', 'Y', 'Y', 'Y',
              'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'n', 'Y', 'Y', 'Y'],
    'throws': ['Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', '', 'Y',
               'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', '', 'Y', 'Y',
               'Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y',
               'Y', 'Y', 'DH', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'N'],
    'runs': [2, 1, 2, 0, 1, 2, 1, 0, 3, 1,
             2, 1, 2, 0, 1, 2, 1, 3, 2, 1,
             2, 0, 1, 2, -1, 1, 2, 1, 0, 2,
             1, 2, 4, 1, 0, 2, -1, 1, 2, 0],
    'descriptions': ['Sharp stop', 'Diving stop', 'Good field', 'Saved boundary', 'Quick throw'] * 8
}

# Kuldeep Yadav - 40 fielding events
kuldeep_data = {
    'balls': 40,
    'positions': ['Short Fine Leg', 'Fine Leg', 'Third Man', 'Deep Fine Leg'] * 10,
    'picks': ['Y', 'Y', 'Y', 'Y', 'n', 'Y', 'Y', 'C', 'Y', 'Y',
              'Y', 'Y', 'Y', 'n', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y',
              'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'n', 'Y', 'DC', 'Y',
              'Y', 'Y', 'Y', 'Y', 'Y', 'C', 'Y', 'Y', 'Y', 'Y'],
    'throws': ['Y', 'N', 'Y', 'N', 'N', 'Y', 'N', '', 'Y', 'N',
               'Y', 'N', 'DH', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y',
               'Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y', '', 'N',
               'Y', 'N', 'Y', 'Y', 'N', '', 'Y', 'N', 'Y', 'Y'],
    'runs': [1, 0, 1, 0, -2, 1, 0, 3, 1, 0,
             1, 0, 4, -1, 1, 0, 1, 2, 0, 1,
             1, 0, 1, 0, 1, 2, -1, 1, -3, 0,
             1, 0, 1, 1, 0, 3, 1, 0, 1, 1],
    'descriptions': ['Boundary save', 'Good chase', 'Quick return', 'Stopped runs', 'Misfield'] * 8
}

# Populate the dataset
over_ball = 0.1
for player_name, player_data in [('Rilee Russouw', russouw_data), 
                                   ('Axar Patel', axar_data), 
                                   ('Kuldeep Yadav', kuldeep_data)]:
    for i in range(player_data['balls']):
        fielding_data['Match_No'].append('IPL2367')
        fielding_data['Innings'].append(1)
        fielding_data['Team'].append('Delhi Capitals')
        fielding_data['Player_Name'].append(player_name)
        
        # Calculate ball count (over.ball format like 0.1, 0.2, ..., 1.1, 1.2, etc.)
        over_num = i // 6
        ball_num = (i % 6) + 1
        fielding_data['Ball_Count'].append(f"{over_num}.{ball_num}")
        fielding_data['Over_Count'].append(over_num + 1)
        
        fielding_data['Position'].append(player_data['positions'][i])
        fielding_data['Short_Description'].append(player_data['descriptions'][i])
        fielding_data['Pick'].append(player_data['picks'][i])
        fielding_data['Throw'].append(player_data['throws'][i])
        fielding_data['Runs'].append(player_data['runs'][i])
        fielding_data['Venue'].append('Arun Jaitley Stadium, Delhi')

# Create DataFrame from the dictionary
df = pd.DataFrame(fielding_data)

print("✓ Dataset created successfully!")
print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Total fielding events recorded: {df.shape[0]}")
print(f"Players analyzed: {df['Player_Name'].unique().tolist()}")
print("\n" + "="*80)
print("FIRST 10 ROWS OF DATA:")
print("="*80)
print(df.head(10).to_string(index=False))

# ============================================================================
# STEP 3: DATA EXPLORATION AND CLEANING
# ============================================================================
# Explanation: Understanding our data before analysis

print("\n" + "="*80)
print("DATA EXPLORATION")
print("="*80)

# Check for missing values
print("\n1. MISSING VALUES CHECK:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "   No missing values found!")

# Check data types
print("\n2. DATA TYPES:")
print(df.dtypes)

# Basic statistics for runs
print("\n3. RUNS SAVED/CONCEDED STATISTICS:")
print(df['Runs'].describe())

# Count fielding events per player
print("\n4. FIELDING EVENTS PER PLAYER:")
print(df['Player_Name'].value_counts())

# Unique positions
print("\n5. FIELDING POSITIONS USED:")
print(f"   Total unique positions: {df['Position'].nunique()}")
for pos in df['Position'].unique():
    print(f"   - {pos}")

# ============================================================================
# STEP 4: DEFINE PERFORMANCE WEIGHTS (AS PER ASSIGNMENT)
# ============================================================================
# Explanation: These are the official weights from your assignment document

weights = {
    'Clean Pick': 1,       # CP: Clean Picks worth 1 point
    'Good Throw': 1,       # GT: Good Throws worth 1 point
    'Catch': 3,            # C: Catches worth 3 points
    'Drop Catch': -3,      # DC: Dropped Catches subtract 3 points
    'Stumping': 3,         # ST: Stumpings worth 3 points
    'Run Out': 3,          # RO: Run Outs worth 3 points
    'Missed Run Out': -2,  # MRO: Missed Run Outs subtract 2 points
    'Direct Hit': 2        # DH: Direct Hits worth 2 points
    # RS: Runs Saved are added as-is (no weight multiplier)
}

print("\n" + "="*80)
print("PERFORMANCE WEIGHTS (OFFICIAL FORMULA)")
print("="*80)
print("\nPS = (CP×WCP) + (GT×WGT) + (C×WC) + (DC×WDC) + (ST×WST) +")
print("     (RO×WRO) + (MRO×WMRO) + (DH×WDH) + RS\n")
for action, weight in weights.items():
    sign = "+" if weight >= 0 else ""
    print(f"   {action:20s}: {sign}{weight:2d} point{'s' if abs(weight) != 1 else ''}")

# ============================================================================
# STEP 5: CALCULATE PERFORMANCE SCORES
# ============================================================================
# Explanation: We analyze each player's performance using the official formula

def calculate_player_performance(player_df):
    """
    Calculate comprehensive performance score for a player using official formula.
    
    This function:
    1. Counts all fielding actions (picks, throws, catches)
    2. Applies the weighted scoring system
    3. Adds runs saved/conceded
    4. Returns detailed statistics and final Performance Score (PS)
    
    Parameters:
    -----------
    player_df : DataFrame
        Contains all fielding records for one player
    
    Returns:
    --------
    dict : Dictionary with all statistics and performance score
    """
    
    # Initialize counters for each type of fielding action
    stats = {
        'clean_picks': 0,
        'good_throws': 0,
        'catches': 0,
        'drop_catches': 0,
        'stumpings': 0,
        'run_outs': 0,
        'missed_run_outs': 0,
        'direct_hits': 0,
        'fumbles': 0,
        'bad_throws': 0,
        'total_runs_saved': 0,
        'total_events': len(player_df)
    }
    
    # Loop through each fielding event for this player
    for _, row in player_df.iterrows():
        pick = str(row['Pick']).strip().upper()  # Get pick type and standardize
        throw = str(row['Throw']).strip().upper()  # Get throw type and standardize
        runs = row['Runs']  # Get runs saved/conceded
        
        # COUNT PICK TYPES
        # Y = Clean Pick, n = Fumble, C = Catch, DC = Drop Catch
        if pick == 'Y':
            stats['clean_picks'] += 1
        elif pick == 'N':
            stats['fumbles'] += 1
        elif pick == 'C':
            stats['catches'] += 1
        elif pick == 'DC':
            stats['drop_catches'] += 1
            
        # COUNT THROW TYPES
        # Y = Good Throw, N = Bad Throw, DH = Direct Hit
        # Additional: RO = Run Out, MR = Missed Run Out, S = Stumping
        if throw == 'Y':
            stats['good_throws'] += 1
        elif throw == 'N':
            stats['bad_throws'] += 1
        elif throw == 'DH':
            stats['direct_hits'] += 1
        elif throw == 'RO':
            stats['run_outs'] += 1
        elif throw == 'MR':
            stats['missed_run_outs'] += 1
        elif throw == 'S':
            stats['stumpings'] += 1
            
        # SUM UP RUNS (positive for saved, negative for conceded)
        stats['total_runs_saved'] += runs
    
    # CALCULATE PERFORMANCE SCORE USING THE OFFICIAL FORMULA
    # PS = (CP×1) + (GT×1) + (C×3) + (DC×-3) + (ST×3) + (RO×3) + (MRO×-2) + (DH×2) + RS
    
    ps = (stats['clean_picks'] * weights['Clean Pick'] +
          stats['good_throws'] * weights['Good Throw'] +
          stats['catches'] * weights['Catch'] +
          stats['drop_catches'] * weights['Drop Catch'] +
          stats['stumpings'] * weights['Stumping'] +
          stats['run_outs'] * weights['Run Out'] +
          stats['missed_run_outs'] * weights['Missed Run Out'] +
          stats['direct_hits'] * weights['Direct Hit'] +
          stats['total_runs_saved'])  # Runs are added as-is
    
    stats['performance_score'] = ps
    
    # Calculate efficiency metrics
    stats['runs_per_event'] = stats['total_runs_saved'] / stats['total_events'] if stats['total_events'] > 0 else 0
    stats['positive_actions'] = stats['clean_picks'] + stats['good_throws'] + stats['catches'] + stats['direct_hits']
    stats['negative_actions'] = stats['fumbles'] + stats['bad_throws'] + stats['drop_catches']
    
    return stats

# Calculate performance for each player
players = df['Player_Name'].unique()
performance_results = {}

print("\n" + "="*80)
print("DETAILED PERFORMANCE ANALYSIS")
print("="*80)

for player in players:
    # Filter data for current player
    player_data = df[df['Player_Name'] == player]
    
    # Calculate their statistics
    performance_results[player] = calculate_player_performance(player_data)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"PLAYER: {player}")
    print(f"{'='*80}")
    print(f"  Total Fielding Events: {performance_results[player]['total_events']}")
    print(f"\n  POSITIVE ACTIONS:")
    print(f"    • Clean Picks (CP):  {performance_results[player]['clean_picks']}")
    print(f"    • Good Throws (GT):  {performance_results[player]['good_throws']}")
    print(f"    • Catches (C):       {performance_results[player]['catches']}")
    print(f"    • Direct Hits (DH):  {performance_results[player]['direct_hits']}")
    print(f"    • Run Outs (RO):     {performance_results[player]['run_outs']}")
    print(f"    • Stumpings (ST):    {performance_results[player]['stumpings']}")
    
    print(f"\n  NEGATIVE ACTIONS:")
    print(f"    • Fumbles:           {performance_results[player]['fumbles']}")
    print(f"    • Bad Throws:        {performance_results[player]['bad_throws']}")
    print(f"    • Drop Catches (DC): {performance_results[player]['drop_catches']}")
    print(f"    • Missed Run Outs:   {performance_results[player]['missed_run_outs']}")
    
    print(f"\n  RUNS IMPACT:")
    print(f"    • Total Runs Saved:  {performance_results[player]['total_runs_saved']:+d}")
    print(f"    • Avg Runs/Event:    {performance_results[player]['runs_per_event']:+.2f}")
    
    print(f"\n  ★ PERFORMANCE SCORE (PS): {performance_results[player]['performance_score']:.1f}")
    
    # Show calculation breakdown
    stats = performance_results[player]
    print(f"\n  CALCULATION BREAKDOWN:")
    print(f"    PS = ({stats['clean_picks']}×1) + ({stats['good_throws']}×1) + ({stats['catches']}×3) + ")
    print(f"         ({stats['drop_catches']}×-3) + ({stats['stumpings']}×3) + ({stats['run_outs']}×3) + ")
    print(f"         ({stats['missed_run_outs']}×-2) + ({stats['direct_hits']}×2) + {stats['total_runs_saved']}")
    print(f"    PS = {stats['clean_picks']} + {stats['good_throws']} + {stats['catches']*3} + ")
    print(f"         {stats['drop_catches']*-3} + {stats['stumpings']*3} + {stats['run_outs']*3} + ")
    print(f"         {stats['missed_run_outs']*-2} + {stats['direct_hits']*2} + {stats['total_runs_saved']}")
    print(f"    PS = {stats['performance_score']:.1f}")

# ============================================================================
# STEP 6: CREATE PERFORMANCE SUMMARY TABLE
# ============================================================================
# Explanation: Organize all results into an easy-to-read table

summary_data = []
for player, stats in performance_results.items():
    summary_data.append({
        'Player': player,
        'Events': stats['total_events'],
        'Clean Picks': stats['clean_picks'],
        'Good Throws': stats['good_throws'],
        'Catches': stats['catches'],
        'Direct Hits': stats['direct_hits'],
        'Run Outs': stats['run_outs'],
        'Drop Catches': stats['drop_catches'],
        'Runs Saved': stats['total_runs_saved'],
        'Performance Score': stats['performance_score']
    })

# Create DataFrame and sort by performance score
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Performance Score', ascending=False)

print("\n" + "="*80)
print("PERFORMANCE SUMMARY TABLE")
print("="*80)
print(summary_df.to_string(index=False))

# ============================================================================
# STEP 7: DATA VISUALIZATIONS
# ============================================================================
# Explanation: Create charts to visualize the analysis results

# VISUALIZATION 1: Overall Performance Score Comparison
# This bar chart shows which player performed best overall
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS...")
print("="*80)

plt.figure(figsize=(12, 7))
colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
bars = plt.bar(summary_df['Player'], summary_df['Performance Score'], color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on top of each bar
for i, (player, score) in enumerate(zip(summary_df['Player'], summary_df['Performance Score'])):
    plt.text(i, score + 2, f'{score:.1f}', ha='center', fontweight='bold', fontsize=14)

plt.xlabel('Player Name', fontsize=13, fontweight='bold')
plt.ylabel('Performance Score (PS)', fontsize=13, fontweight='bold')
plt.title('IPL Fielding Performance Comparison\nDelhi Capitals - Match IPL2367', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=15, ha='right', fontsize=11)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.ylim(0, max(summary_df['Performance Score']) * 1.15)

# Add a horizontal line for average
avg_score = summary_df['Performance Score'].mean()
plt.axhline(y=avg_score, color='orange', linestyle='--', linewidth=2, label=f'Team Average: {avg_score:.1f}')
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('1_performance_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization 1 saved: 1_performance_comparison.png")
plt.close()

# VISUALIZATION 2: Fielding Actions Breakdown (Stacked Bar Chart)
# Shows composition of each player's fielding activities

fig, ax = plt.subplots(figsize=(14, 7))

players_list = summary_df['Player'].tolist()
clean_picks = summary_df['Clean Picks'].tolist()
good_throws = summary_df['Good Throws'].tolist()
catches = summary_df['Catches'].tolist()
direct_hits = summary_df['Direct Hits'].tolist()

x = np.arange(len(players_list))
width = 0.6

# Create stacked bars
p1 = ax.bar(x, clean_picks, width, label='Clean Picks', color='#3498db')
p2 = ax.bar(x, good_throws, width, bottom=clean_picks, label='Good Throws', color='#2ecc71')
p3 = ax.bar(x, catches, width, bottom=np.array(clean_picks)+np.array(good_throws), 
            label='Catches', color='#e74c3c')
p4 = ax.bar(x, direct_hits, width, 
            bottom=np.array(clean_picks)+np.array(good_throws)+np.array(catches),
            label='Direct Hits', color='#f39c12')

ax.set_xlabel('Player Name', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Actions', fontsize=13, fontweight='bold')
ax.set_title('Breakdown of Fielding Actions by Player\nIPL2367 - Delhi Capitals', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(players_list, rotation=15, ha='right')
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('2_actions_breakdown_stacked.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 2 saved: 2_actions_breakdown_stacked.png")
plt.close()

# VISUALIZATION 3: Runs Saved Comparison with Positive/Negative Split
# Shows the runs impact of each player

plt.figure(figsize=(12, 7))
colors_runs = ['#27ae60' if x >= 0 else '#e74c3c' for x in summary_df['Runs Saved']]
bars = plt.bar(summary_df['Player'], summary_df['Runs Saved'], color=colors_runs, 
               edgecolor='black', linewidth=1.5)

# Add value labels
for i, (player, runs) in enumerate(zip(summary_df['Player'], summary_df['Runs Saved'])):
    y_pos = runs + (1 if runs >= 0 else -1)
    plt.text(i, y_pos, f'{runs:+d}', ha='center', fontweight='bold', fontsize=13)

plt.xlabel('Player Name', fontsize=13, fontweight='bold')
plt.ylabel('Total Runs Saved/Conceded', fontsize=13, fontweight='bold')
plt.title('Runs Impact Through Fielding\nIPL2367 - Delhi Capitals', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=15, ha='right', fontsize=11)
plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#27ae60', label='Runs Saved (Positive)'),
                   Patch(facecolor='#e74c3c', label='Runs Conceded (Negative)')]
plt.legend(handles=legend_elements, loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('3_runs_saved_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 3 saved: 3_runs_saved_comparison.png")
plt.close()

# VISUALIZATION 4: Fielding Position Heatmap
# Shows which positions each player fields in and their effectiveness

position_performance = df.groupby(['Player_Name', 'Position']).agg({
    'Runs': 'sum'  # Total runs saved at each position
}).reset_index()

# Create pivot table for heatmap
position_pivot = position_performance.pivot(index='Player_Name', 
                                             columns='Position', 
                                             values='Runs').fillna(0)

plt.figure(figsize=(14, 6))
sns.heatmap(position_pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=0, 
            linewidths=2, linecolor='white', cbar_kws={'label': 'Runs Saved'},
            vmin=-10, vmax=30)

plt.xlabel('Fielding Position', fontsize=13, fontweight='bold')
plt.ylabel('Player Name', fontsize=13, fontweight='bold')
plt.title('Runs Saved by Player and Fielding Position\nIPL2367 - Delhi Capitals', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()
plt.savefig('4_position_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 4 saved: 4_position_heatmap.png")
plt.close()

# VISUALIZATION 5: Positive vs Negative Actions Comparison
# Shows the ratio of good to bad fielding actions

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(players_list))
width = 0.35

positive = [performance_results[p]['positive_actions'] for p in players_list]
negative = [performance_results[p]['negative_actions'] for p in players_list]

bars1 = ax.bar(x - width/2, positive, width, label='Positive Actions', color='#2ecc71', edgecolor='black')
bars2 = ax.bar(x + width/2, negative, width, label='Negative Actions', color='#e74c3c', edgecolor='black')

# Add value labels
for i, v in enumerate(positive):
    ax.text(i - width/2, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=11)
for i, v in enumerate(negative):
    ax.text(i + width/2, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=11)

ax.set_xlabel('Player Name', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Actions', fontsize=13, fontweight='bold')
ax.set_title()