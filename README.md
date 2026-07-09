->This program helps us work on a real world dataset taken from kaggle. The data set provides use information about Delhi's traffic on how much time is taken to travel
from one destination to another given under various circumstances.
->In the code, pandas are used to read the data from the given csv, then networkx library is used to make a graph where each area is given a node number starting
from 0.
-> All the node values are stored in a hashtable.
-> Based on that graphs are built in the program and and we run Dijkstra's algorithm to find out the smallest path from source to destination entered by the user.
-> Frontend is made using streamlit.
