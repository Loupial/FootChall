#!/usr/bin/env python

#Generate simulation of Football Games using Footbar templates

import argparse
import concurrent.futures
import json
import os
import pandas as pd
import numpy as np


def combine_json_files_in_folder(folder_path):
    """
    Combine all .json files in a folder into a single DataFrame.

    Args:
        folder_path (str): The path to the folder containing .json files.

    Returns:
        pd.DataFrame: A DataFrame containing the combined data from all .json files.
    """
    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # List all files in the specified folder
    files = os.listdir(folder_path)

    # Iterate through the files and combine them
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_json(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

def get_transition(full_df):
    """
    Calculate transition probabilities between action labels in a DataFrame.

    Args:
        full_df (pd.DataFrame): The DataFrame containing action labels.

    Returns:
        pd.DataFrame: A DataFrame representing the transition probabilities between action labels.
    """
    labels_game = full_df.label
    label_data = {}

    # Count occurrences and followed counts
    for i in range(len(labels_game) - 1):
        current_label = labels_game[i]
        next_label = labels_game[i + 1]

        if current_label not in label_data:
            label_data[current_label] = {'total': 0, 'followed': {}}

        label_data[current_label]['total'] += 1

        if next_label not in label_data[current_label]['followed']:
            label_data[current_label]['followed'][next_label] = 0

        label_data[current_label]['followed'][next_label] += 1

    # Convert the nested dictionary into a DataFrame
    data_for_df = {}

    for label, data in label_data.items():
        percentages = {followed_label: (followed_count / data['total']) if data['total'] > 0 else 0
                       for followed_label, followed_count in data['followed'].items()}
        data_for_df[label] = percentages

    df_per = pd.DataFrame(data_for_df).fillna(0).T
    return df_per

def Del_no_action(df_per):
    """
    Remove 'no action' transitions from the transition probability DataFrame.

    Args:
        df_per (pd.DataFrame): The transition probability DataFrame.

    Returns:
        pd.DataFrame: A modified DataFrame with 'no action' transitions removed.
    """
    for columns in df_per.columns:
        if df_per['no action'][columns] != 0:
            df_per['walk'][columns] = df_per['walk'][columns] + df_per['no action'][columns]
    
    # Drop 'no action' row and column
    df_per = df_per.drop('no action', axis=0)
    df_per = df_per.drop('no action', axis=1)
    
    return df_per

def modify_transition_matrices(transition_matrix):
    """
    Modify and normalize transition matrices for attacking and defending scenarios.

    Args:
        transition_matrix (dict): A dictionary containing transition probabilities.

    Returns:
        pd.DataFrame: Two modified and normalized DataFrames for attacking and defending scenarios.
    """
    # Create a DataFrame from the provided transition_matrix
    transition_df = pd.DataFrame(transition_matrix, index=['walk', 'rest', 'run', 'tackle', 'dribble', 'pass', 'cross', 'shot']).T

    # Create two copies of the transition_df for attacking and defending scenarios
    transition_attacking = transition_df.copy()
    transition_defending = transition_df.copy()

    # Modify probabilities in the attacking dataframe
    transition_attacking['run']['pass'] += 0.15
    transition_attacking['walk']['pass'] += 0.1
    transition_attacking['shot']['pass'] += 0.05

    transition_attacking['run']['run'] += 0.1
    transition_attacking['pass']['run'] += 0.1
    transition_attacking['dribble']['run'] += 0.05
    transition_attacking['shot']['run'] += 0.05

    transition_attacking['run']['dribble'] += 0.1
    transition_attacking['pass']['dribble'] += 0.1
    transition_attacking['shot']['dribble'] += 0.05

    # Modify probabilities in the defending dataframe
    transition_defending['walk']['walk'] += 0.1
    transition_defending['run']['walk'] += 0.1
    transition_defending['rest']['walk'] += 0.1

    transition_defending['run']['rest'] += 0.1
    transition_defending['walk']['rest'] += 0.1

    transition_defending['run']['tackle'] += 0.1
    transition_defending['rest']['tackle'] += 0.1
    transition_defending['walk']['tackle'] += 0.1

    # Normalize the probabilities in both DataFrames to ensure they sum to 1
    transition_attacking = transition_attacking.div(transition_attacking.sum(axis=1), axis=0)
    transition_defending = transition_defending.div(transition_defending.sum(axis=1), axis=0)

    return transition_attacking, transition_defending

def stats_df(full_df):
    """
    Calculate standard deviation for different action labels in a DataFrame.

    Args:
        full_df (pd.DataFrame): The DataFrame containing action labels and corresponding time series data.

    Returns:
        dict: A dictionary containing standard deviations for each action label.
    """
    actions = {}
    label_list = ['walk', 'rest', 'run', 'tackle', 'dribble', 'pass', 'cross', 'shot', 'no action']

    for label in label_list:
        lab_df = full_df.loc[full_df['label'] == label, 'norm'].explode()

        # Calculate the standard deviation directly
        std_deviation = lab_df.std()

        actions[label] = {
            'std': std_deviation
        }

    return actions

def SimGames(transition_matrix,transition_attacking, transition_defending,Time,Type):
    """
    Simulate a game using Markov Chain Monte Carlo (MCMC) based on the specified type.

    Args:
        Input_folder (str) : Path to the folder where you have the original json data (match_1 and match_2, if you have more data the generation of games will be better).
        Time (int): The duration of the game in minutes.
        Type (str): The type of the game, which can be "Attacking", "Defending", or "Normal".

    Returns:
        list: A list of simulated game events based on the selected type.
    """

    # Number of steps in the MCMC simulation
    num_steps = int(Time*60*50/46) #46 is the average length of a serie

    # Initial state (replace with the initial label index)
    current_state = np.random.choice([0, 1, 2, 5])

    # Create an array to store the samples
    samples = []
    if (Type == "Attacking") or (Type == "attacking") :
        print(" You chose a Attacking Game simulation")
        df_per = transition_attacking
    elif (Type == "Defending") or (Type == "defending"):
        print(" You chose a Defending Game simulation")
        df_per = transition_defending
    else : 
        df_per = transition_matrix
    # Perform the MCMC simulation
    for _ in range(num_steps):
        # Propose a new state based on the transition probabilities
        proposed_state = np.random.choice(len(df_per), p=df_per.iloc[current_state])
        
        # Move to the proposed state
        current_state = proposed_state
        
        # Store the current state as a sample
        samples.append(current_state)

    # Convert label indices to actual labels
    label_mapping = ['walk', 'rest', 'run', 'tackle', 'dribble', 'pass', 'shot', 'cross']
    sampled_labels = [label_mapping[state] for state in samples]
    return sampled_labels

def simulate_random_walk(Random_pick, step_std, max_iterations=1000):
  
    """
    Simulate a random walk with noise.

    Args:
        Random_pick (list): The initial data points for the random walk.
        step_std (float): The standard deviation of the random steps.
        max_iterations (int, optional): The maximum number of iterations (default: 1000).

    Returns:
        list: A list representing the simulated random walk data.
              If the simulation fails, it returns a list with a single element [-1].
    """
      
    noisy_series = Random_pick.copy()  # Create a copy to avoid modifying the original

    for _ in range(max_iterations):
        random_steps = np.random.normal(loc=0, scale=step_std, size=len(Random_pick))
        small_positive_constant = 0.15 * max(Random_pick)
        noisy_series += np.cumsum(random_steps)
        negative_indices = np.where(noisy_series < 5)

        noisy_series[negative_indices] += small_positive_constant
        # Adjust as needed
        if not any(element < 0 for element in noisy_series):
            return noisy_series
    return [-1]  # Return the current state even if max iterations are reached

def simulate_play(actions,play, attempt=0):
    """
    Simulate a random walk for a given play using the games from the input_folder as a database were we will randomly pick the play.

    Args:
        Input_folder (str) : Path to the folder where you have the original json data (match_1 and match_2, if you have more data the generation of games will be better).
        play (str): The type of play (e.g., 'walk', 'rest', 'run').
        attempt (int): The current attempt count for recursion (default: 0).

    Returns:
        list: A list representing the simulated random walk data.
              If an error occurs or the simulation fails, it returns an empty list.
    """

    

    if attempt >= 100:
        # Stop recursion if it exceeds 100 attempts
        return []

    Random_pick = dataframes.loc[dataframes['label'] == play, 'norm'].sample().values[0]
    step_std = actions[play]['std'] * 0.1

    # Simulate random walk for the play
    noisy_series = simulate_random_walk(Random_pick, step_std)
    
    if noisy_series[0] != -1:
        return noisy_series
    else:
        # Retry the simulation with recursion
        return simulate_play(play, attempt + 1)

def add_transition_values(series_list):
    """
    Add a point at the end of each series in the list, equal to a random number between ±10% of the first point of the next series.

    Args:
        series_list (list): A list of time series data.
    """

    new_series_list = []  # Create a new list to store the modified series
    for i in range(len(series_list) - 1):
        current_series = series_list[i]
        next_series = series_list[i + 1]
        transition_value = np.random.uniform(0.9, 1.1)*next_series[0]
        new_series = np.append(current_series,transition_value)
        # Add the transition value to the end of the current series
        new_series_list.append(new_series)
    new_series_list.append(series_list[-1])
    return new_series_list

def dataframe_to_json(dataframe, output_file):
    """
    Convert a DataFrame to a JSON file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be converted.
        output_file (str): The path to the output JSON file.
    """
    dataframe['norm'] = dataframe['norm'].apply(lambda x: x.tolist())
    # Convert the DataFrame to a list of dictionaries
    data_as_dict = dataframe.to_dict(orient='records')

    # Write the list of dictionaries to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(data_as_dict, json_file)

def Generate_game(dataframes,Number_games,Time,Mode,output_folder_path,num_workers=8):

    if isinstance(Mode, str):
        # If Mode is a string, convert it to a list with the same value for each game
        Mode = [Mode] * Number_games
    elif isinstance(Mode, list):
        # If Mode is a string, convert it to a list with the same value for each game
        if len(Mode) == 1:
            Mode = [Mode] * Number_games
            
        elif len(Mode) != Number_games:
        # If Mode is a list, ensure it has the same number of elements as Number_games
            raise ValueError("Number of elements in 'Mode' list must be equal to 'Number_games'")
    else:
        raise ValueError("'Mode' must be a string or a list of strings")
    
    transition_matrix = Del_no_action(get_transition(dataframes))
    transition_attacking, transition_defending = modify_transition_matrices(transition_matrix)
    actions = stats_df(dataframes)

    for i in range (0,Number_games):
        if (str(Mode[i]) == '["\'Attacking\'"]' or str(Mode[i])=='["\'Normal\'"]' or str(Mode[i])=='["\'Defending\'"]'):
            call = str(Mode[i][0])
        else : call = str(Mode[i])
        file_path=output_folder_path + "\Game_"+call+"_"+str(i)+'.json'
        Label_generated = SimGames(transition_matrix, transition_attacking, transition_defending, Time, Mode[i])

        # List to store simulated game data
        simulated_game = []
        
        # Create a ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Simulate random walks for each play in parallel
            arg_pairs = [(actions, play) for play in Label_generated]

            # Use executor.map with the list of arg_pairs
            results = list(executor.map(lambda args: simulate_play(*args), arg_pairs))
        # Store the results in the simulated_game list
        simulated_game.extend(results)
        
        df_generated = pd.DataFrame({'label': Label_generated, 'norm': add_transition_values(simulated_game)})
        dataframe_to_json(df_generated,file_path)

    return print('Your '+str(Number_games) +' simulated games were saved here : '+ file_path)

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Simulate and save games.')
    
    # Add arguments
    
    parser.add_argument('-n','--Number_games', type=int, required=True, help='Number of games you want to simulate')
    parser.add_argument('-t','--Time', type=int, required=True, help='Duration of the games you want to simulate')
    parser.add_argument('-m','--Mode', nargs='+', default="Normal", help='Mode of game can either be "Attacking", "Defending" or "Normal" (str if you want all games to be the same mode or a list with the same size as the number of gaes you want to simulate)')
    parser.add_argument('-i','--Input_folder', type=str, default="Data", help='Name of the folder containing .json files (relative or absolute path) where you have the original data (match_1 and match_2, if you have more data the generation of games will be better)')
    parser.add_argument('-o','--output_folder_path', type=str, default="GamesGenerated", help='Path to the output folder (optional)')
    parser.add_argument('-w','--num_workers', type=int, default=4, help='Number of worker threads (default: 4)')
    
    # Parse arguments
    args = parser.parse_args()

    try:
        # Call the Generate_game function with the specified arguments
        
        if os.path.isabs(args.Input_folder):
            folder_path = args.Input_folder  # Treat as an absolute path
        else:
            # Combine the provided folder name with the current working directory to get the full path
            folder_path = os.path.join(os.getcwd(), args.Input_folder)

        if args.output_folder_path is None:
            output_folder_path = os.path.join(os.getcwd(), "GameGenerated")
        else:
            if os.path.isabs(args.output_folder_path):
                output_folder_path = args.output_folder_path  # Treat as an absolute path
            else:
                # Combine the provided folder name with the current working directory to get the full path
                output_folder_path = os.path.join(os.getcwd(), args.output_folder_path)

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder_path, exist_ok=True)
        dataframes = combine_json_files_in_folder(folder_path)

        Generate_game(dataframes, args.Number_games, args.Time, args.Mode, output_folder_path, args.num_workers)
        print('Simulation completed successfully.')
    except Exception as e:
        print(f'Error: {e}')