# FootChall

A Football datascience Challenge : Generative AI recreating football game

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Credits](#credits)
- [Support](#support)

## Installation

Download the project. 
**Make sure that you put the data inside the "Data" folder.**
For the exercise we only used the 2 games provided, but you can always put as many as you want. 
Using the command go to the directory where you downloaded the script, and run it.


## Usage

2 files : 

* Footbar.ipynb : The Jupyter notebook where you can find the analytics and the explanations.

* Simulations.py : The Python script containing the code to generate as may game as you want, that can be run as a one-liner.



Examples : python Simulations.py --Number_games 1 --Time 10 --Mode 'Defending'

**Arguments:**

+ '-n'/ '--Number_games' / type=int / required=True / **Number of games you want to simulate**

+ '-t' / '--Time' / type=int / required=True / **Duration of the games you want to simulate'**

+ '-m' / '--Mode' / nargs='+' / default="Normal" / **Mode of game can either be "Attacking", "Defending" or "Normal" (str if you want all games to be the same mode or a list with the same size as the number of gaes you want to simulate)**

+ '-i' / '--Input_folder' / type=str / default="Data" / **Name of the folder containing .json files (relative or absolute path) where you have the original data (match_1 and match_2, if you have more data the generation of games will be better)**

+ '-o' / '--output_folder_path' / type=str / default="GamesGenerated", **Path to the output folder (relative or absolute)**

+ '-w' / '--num_workers' / type=int / default=4 / **Number of worker threads**


## Configuration

Requirements.txt :  
+ numpy==1.23.5
+ pandas==1.5.3


## Credits

Made by Jérémy Boussaguet


## Support

For questions or support, contact me at jeremy.boussaguet@pm.me
