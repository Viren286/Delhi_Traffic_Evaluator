# Delhi Traffic Route Optimizer

This project uses a real-world traffic dataset from Kaggle to determine the shortest travel route between two locations in Delhi under different traffic conditions.

## Features

* Uses a real-world Delhi traffic dataset from Kaggle.
* Builds a weighted graph representing locations and travel times.
* Computes the shortest path between a source and destination using **Dijkstra's Algorithm**.
* Provides an interactive web interface built with **Streamlit**.

## How It Works

1. **Data Processing**

   * The traffic dataset is loaded from a CSV file using the **Pandas** library.
   * Each unique location is assigned a unique node ID starting from **0**.
   * A hash map (dictionary) is maintained to efficiently map location names to their corresponding node IDs.

2. **Graph Construction**

   * The **NetworkX** library is used to construct a weighted graph.
   * Nodes represent different locations in Delhi.
   * Edges represent roads between locations, with edge weights corresponding to the estimated travel time from the dataset.

3. **Shortest Path Calculation**

   * The user selects a source and destination through the Streamlit interface.
   * The application runs **Dijkstra's Algorithm** on the graph to compute the minimum travel-time path.
   * The shortest route and total travel time are displayed to the user.

4. **User Interface**

   * The frontend is developed using **Streamlit**, providing a simple and interactive interface for selecting locations and viewing the optimal route.

## Technologies Used

* Python
* Pandas
* NetworkX
* Streamlit
* Dijkstra's Algorithm

## Dataset

The project uses a Delhi traffic dataset obtained from **Kaggle**, containing travel-time information between locations under various traffic conditions. This data is used to construct the weighted graph on which shortest-path computations are performed.
