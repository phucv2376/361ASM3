import signal
import pandas as pd
from collections import defaultdict, deque
import heapq
import math
import time

# https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad


    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in kilometers

# LLMP PROMPT: define a weighted bidirectional graph structure for a city and adjacent cities
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)  # {city: [(neighbor, distance), ...]}
        self.city_locations = {}

    def add_edge(self, city1, city2, distance):
        self.graph[city1].append((city2, distance))
        self.graph[city2].append((city1, distance))  # Ensure bidirectional connection

    def load_cities(self, filename):
        # Load cities and their coordinates using pandas dataframe
        df_cities = pd.read_csv(filename, header=None, names=['city', 'latitude', 'longitude'])
        self.city_locations = {row['city']: (row['latitude'], row['longitude']) for index, row in df_cities.iterrows()}

    def load_adjacencies(self, filename):
        # Load adjacencies using pandas dataframe (whitespace-delimited)
        df_adjacencies = pd.read_csv(filename, sep="\s+", header=None, names=['city1', 'city2'])

        # Get weight of graph using haversine; weight = distance
        for index, row in df_adjacencies.iterrows():
            city1, city2 = row['city1'], row['city2']
            if city1 in self.city_locations and city2 in self.city_locations:
                lat1, lon1 = self.city_locations[city1]
                lat2, lon2 = self.city_locations[city2]
                distance = haversine(lat1, lon1, lat2, lon2)
                self.add_edge(city1, city2, distance)

# LLM PROMPT: define a bfs function that has graph, start, and goal parameters. return path after performing bfs
def bfs(graph, start, goal):
    visited = set()
    priority_queue = [(0, start, [start])]  # (cumulative_distance, current_city, path)

    while priority_queue:
        current_distance, current_city, path = heapq.heappop(priority_queue)
        if current_city == goal:
            return path  # Return the found path
        if current_city not in visited:
            visited.add(current_city)
            for neighbor, distance in graph.graph[current_city]:
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (current_distance + distance, neighbor, path + [neighbor]))
    return None  # No path found

#LLM PROMPT: define a dfs function that has graph, start, and goal parameters. return path after performing dfs
def dfs(graph, start, goal):
    visited = set()
    stack = [(0, start, [start])]  # (cumulative_distance, current_city, path)

    while stack:
        current_distance, current_city, path = stack.pop()
        if current_city == goal:
            return path  # Return the found path
        if current_city not in visited:
            visited.add(current_city)
            for neighbor, distance in graph.graph[current_city]:
                if neighbor not in visited:
                    stack.append((current_distance + distance, neighbor, path + [neighbor]))
    return None  # No path found

#LLM PROMPT: define a iddfs function that has graph, start, goal, and max_depth parameters. return path after performing iddfs
def iddfs(graph, start, goal, max_depth):
    def dls(current_city, goal, depth, current_distance, path, visited):
        if depth == 0 and current_city == goal:
            return path
        if depth > 0:
            visited.add(current_city)
            for neighbor, distance in graph.graph[current_city]:
                if neighbor not in visited:
                    result = dls(neighbor, goal, depth-1, current_distance + distance, path + [neighbor], visited)
                    if result:
                        return result
            visited.remove(current_city)
        return None

    for depth in range(max_depth):
        visited = set()
        result = dls(start, goal, depth, 0, [start], visited)
        if result:
            return result
    return None

#LLM Generated heuristic as part of Best First and A*;
def heuristic(city1, city2, city_locations):
    lat1, lon1 = city_locations[city1]
    lat2, lon2 = city_locations[city2]
    return haversine(lat1, lon1, lat2, lon2)  # Use Haversine distance

#LLM PROMPT: define a best first search function that takes in a graph, start and goal. return the path by performing best first search
def best_first_search(graph, start, goal):
    visited = set()
    priority_queue = [(0, start, [start])]  # (heuristic + distance, current_city, path)

    while priority_queue:
        _, current_city, path = heapq.heappop(priority_queue)
        if current_city == goal:
            return path
        if current_city not in visited:
            visited.add(current_city)
            for neighbor, distance in graph.graph[current_city]:
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (distance + heuristic(neighbor, goal, graph.city_locations), neighbor, path + [neighbor]))
    return None

#LLM PROMPT: define a* function that takes in a graph, start and goal. return the path using a* search
def a_star_search(graph, start, goal):
    visited = set()
    g_costs = {start: 0}  # Distance from start to current node
    priority_queue = [(heuristic(start, goal, graph.city_locations), start, [start])]  # (f_cost, current_city, path)

    while priority_queue:
        f_cost, current_city, path = heapq.heappop(priority_queue)
        if current_city == goal:
            return path
        if current_city not in visited:
            #f(n) = g(n) + h(n)
            visited.add(current_city)
            for neighbor, distance in graph.graph[current_city]:
                g_cost = g_costs[current_city] + distance  # Add edge weight
                if neighbor not in g_costs or g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = g_cost
                    f_cost = g_cost + heuristic(neighbor, goal, graph.city_locations)
                    heapq.heappush(priority_queue, (f_cost, neighbor, path + [neighbor]))
    return None

# Calculate the total distance of a given path
def calculate_path_distance(graph, path):
    total_distance = 0.0
    for i in range(len(path) - 1):
        city1, city2 = path[i], path[i + 1]
        # Find the distance between consecutive cities
        for neighbor, distance in graph.graph[city1]:
            if neighbor == city2:
                total_distance += distance
                break
    return total_distance

def timeout_handler(signum, frame):
    raise Exception("Search algorithm timed out")

# LLM prompt: create a function that takes in a function, its arguments and a timeout value, the function should run the given function with a timeout
def run_with_timeout(func, args=(), timeout_duration=10):
    """
    Runs the given function with a timeout.
    :param func: The search function to run
    :param args: Arguments to pass to the function
    :param timeout_duration: Maximum allowed duration in seconds
    :return: The result of the function or a timeout exception
    """
    # Set the signal handler for the timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)  # Set the alarm

    try:
        result = func(*args)  # Call the function with its arguments
        signal.alarm(0)  # Reset the alarm after successful execution
        return result
    except Exception as ex:
        print(f"Error: {str(ex)}")
        return None
    finally:
        signal.alarm(0)  # Ensure the alarm is turned off after execution


# Main function for user interaction
g = Graph()
g.load_cities("Coordinates.csv")
g.load_adjacencies("Adjacencies.txt")

while True:
    start = input("Start City: ")
    end = input("Destination City: ")

    if start not in g.city_locations or end not in g.city_locations:
        print("Invalid city names entered. Please try again.")
        continue

    print("Choose a search method:")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Iterative Deepening DFS (ID-DFS)")
    print("4. Best-First Search")
    print("5. A* Search")

    choice = int(input("Enter your choice (1-5): "))

    start_time = time.time()

    if choice == 1:
        path = run_with_timeout(bfs, (g, start, end), timeout_duration=30)
    elif choice == 2:
        path = run_with_timeout(dfs, (g, start, end), timeout_duration=30)
    elif choice == 3:
        path = run_with_timeout(iddfs, (g, start, end, 100), timeout_duration=30)
    elif choice == 4:
        path = run_with_timeout(best_first_search, (g, start, end), timeout_duration=30)
    elif choice == 5:
        path = run_with_timeout(a_star_search, (g, start, end), timeout_duration=30)
    else:
        print("Invalid choice. Please try again.")
        continue

    end_time = time.time()
    elapsed_time = end_time - start_time

    if path:
        print("Path found: ", " -> ".join(path))
        total_distance = calculate_path_distance(g, path)
        print(f"Total path distance: {total_distance:.2f} km")
    else:
        print("No path found.")

    print(f"Search time: {elapsed_time:.4f} seconds")

    is_continue = input("Do you want to try another method? (y/n): ")
    if is_continue.lower() != 'y':
        break