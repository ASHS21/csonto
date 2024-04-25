import json
from neo4j import GraphDatabase
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms import link_prediction
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import py2neo
import community as community_louvain
import scipy 
import re
import fitz  # PyMuPDF, install with 'pip install PyMuPDF'
import pandas as pd
import time

# Neo4j Connection Class
class Neo4jConnection:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.driver = GraphDatabase.driver(
            self.config['neo4j']['uri'],
            auth=(self.config['neo4j']['user'], self.config['neo4j']['password'])
        )

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as file:
            return json.load(file)

    def close(self):
        if self.driver:
            self.driver.close()

    def fetch_graph_data(self):
        with self.driver.session() as session:
            query = """
            
            MATCH (n:CyberSecurityScore)<-[r:REPORTS_TO]-(m) RETURN n, r, m 
            
            """
            results = session.run(query)
            records = list(results)  # Store results in a list to prevent ResultConsumedError
        return records
    
    def fetch_graph_data2(self):
        with self.driver.session() as session:
            # Fetch all distinct nodes regardless of their labels or relationships
            node_query = "MATCH (n) RETURN count(DISTINCT n) AS node_count"
        
            # Fetch all relationships, not just those of type REPORTS_TO
            edge_query = "MATCH ()<-[r]-() RETURN count(DISTINCT r) AS edge_count"
        
            node_count_result = session.run(node_query).single()["node_count"]
            edge_count_result = session.run(edge_query).single()["edge_count"]

            return node_count_result, edge_count_result

neo4j_conn = Neo4jConnection("/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/pages/config.json")

# Function to clean and extract terms from a text block
def extract_terms(text_block):
    return re.findall(r'\b\w+\b', text_block.lower())  # Extract words and convert to lower case

# Extract terms from the PDF
def extract_terms_from_pdf(pdf_path):
    terms = set()
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            terms.update(extract_terms(text))
    return terms

# Load CSV datasets and extract terms
def extract_terms_from_csvs(csv_files):
    terms = set()
    for file in csv_files:
        df = pd.read_csv(file)
        for column in df.columns:
            column_text = " ".join(df[column].astype(str))
            terms.update(extract_terms(column_text))
    return terms

# Function to compare terms from datasets with KG
def compare_terms_with_kg(dataset_terms, kg_terms):
    common_terms = dataset_terms.intersection(kg_terms)
    return {
        "total_terms_from_datasets": len(dataset_terms),
        "total_terms_in_kg": len(kg_terms),
        "common_terms_count": len(common_terms),
        "common_terms_list": list(common_terms)
    }

# Load CSV and extract terms
def extract_terms_from_csv(file_path):
    df = pd.read_csv(file_path)
    terms = set()
    for column in df.columns:
        column_text = " ".join(df[column].astype(str))
        terms.update(extract_terms(column_text))
    return terms



# Calculate metrics
def calculate_metrics(gold_standard_terms, kg_terms):
    true_positives = gold_standard_terms.intersection(kg_terms)
    predicted_positives = len(kg_terms)
    actual_positives = len(gold_standard_terms)
    
    precision = len(true_positives) / predicted_positives if predicted_positives else 0
    recall = len(true_positives) / actual_positives if actual_positives else 0
    coverage = len(true_positives) / actual_positives if actual_positives else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'coverage': coverage,
        'true_positives': true_positives,
        'false_positives': predicted_positives - len(true_positives),
        'false_negatives': actual_positives - len(true_positives),
    }

# Function to load and extract terms from a JSON file
def load_and_extract_terms_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)  # This should correctly parse the JSON into a dictionary
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return set()
    except Exception as e:
        print("An error occurred:", e)
        return set()
    
    return extract_terms_from_json(json_data)

# Function to extract terms from the JSON structure
def extract_terms_from_json(json_data):
    terms = set()
    try:
        results = json_data['results']['bindings']
        for result in results:
            for key, value_obj in result.items():
                if value_obj['type'] == 'literal':
                    terms.update(extract_terms(value_obj['value']))
    except TypeError as e:
        print("TypeError:", e)
        print("Current JSON data structure:", json_data)
    except KeyError as e:
        print("KeyError: Missing a key in JSON data -", e)
    return terms

# Path to your JSON file
json_file_path = "/Users/alialmoharif/Desktop/D3fend Full Mappings.json"

# Extract terms from the gold standard JSON
defend_terms = load_and_extract_terms_from_json(json_file_path)

csv_file_path = '/Users/alialmoharif/Desktop/CSKG-FYP.csv'

# Paths to your files
pdf_path = "/Users/alialmoharif/Desktop/FYP/Dataset/Cybersecurity Acronyms 2020.pdf"
csv_files = [
    "/Users/alialmoharif/Desktop/FYP/Dataset/5.12 Cybersecurity Detail.csv",
    "/Users/alialmoharif/Desktop/FYP/Dataset/Access Log Jan 01 2017.csv",
    "/Users/alialmoharif/Desktop/FYP/Dataset/Cybersecurity Summary.csv"
]

# Extract terms from datasets
terms_from_pdf = extract_terms_from_pdf(pdf_path)
terms_from_csvs = extract_terms_from_csvs(csv_files)
all_dataset_terms = terms_from_pdf.union(terms_from_csvs)

# Placeholder for KG terms - replace with actual extraction from your KG
kg_terms = extract_terms_from_csv(csv_file_path)  # You will need to populate this set with terms from your KG

# Compare and get results
metrics = calculate_metrics(defend_terms, kg_terms)
# Now print the metrics without listing all the true positive terms
print("Gold Standard between the KG and another KG")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"Coverage: {metrics['coverage']:.4f}")
print(f"True Positives Count: {len(metrics['true_positives'])}")
print(f"False Positives Count: {metrics['false_positives']}")
print(f"False Negatives Count: {metrics['false_negatives']}")

print("\n\n")

metrics2 = calculate_metrics(all_dataset_terms, defend_terms)
# Now print the metrics without listing all the true positive terms
print("Gold Standard between the Gold KG and domain datasets")
print(f"Precision: {metrics2['precision']:.4f}")
print(f"Recall: {metrics2['recall']:.4f}")
print(f"Coverage: {metrics2['coverage']:.4f}")
print(f"True Positives Count: {len(metrics2['true_positives'])}")
print(f"False Positives Count: {metrics2['false_positives']}")
print(f"False Negatives Count: {metrics2['false_negatives']}")

print("\n\n")

# Match KG terms with domain terms to find true positives
true_positives = kg_terms.intersection(all_dataset_terms)

# Calculate metrics
predicted_positives = len(kg_terms)  # All KG terms considered as predictions
actual_positives = len(all_dataset_terms)  # All terms from domain datasets considered as actual positives

precision = len(true_positives) / predicted_positives if predicted_positives else 0
recall = len(true_positives) / actual_positives if actual_positives else 0
coverage = len(true_positives) / actual_positives if actual_positives else 0

print("\n\n")
print("\n\n")


# Output results
print(f"Total KG Terms: {predicted_positives}")
print(f"Total Domain Terms from Datasets: {actual_positives}")
print(f"Relevant KG Terms Identified: {len(true_positives)}")
print(f"True Positives: {len(true_positives)}")
print(f"False Positives: {predicted_positives - len(true_positives)}")
print(f"False Negatives: {actual_positives - len(true_positives)}")
print(f"Coverage: {coverage:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Fetch the graph data from Neo4j
records = neo4j_conn.fetch_graph_data()
N, E = neo4j_conn.fetch_graph_data2()

# Calculate sparsity beta
beta = 1 - (E / (N * (N - 1)))  # for directed graph

# Print out the KG properties and the calculated metrics
print(f"Number of nodes (N): {N}")
print(f"Number of edges (E): {E}")
print(f"Sparsity (beta): {beta}")


G = nx.DiGraph()

# Populate the graph with nodes and edges along with labels
for record in records:
    # Get the identity (ID) of the source and target nodes
    source_id = record["n"]._id
    target_id = record["m"]._id
    
    # Use 'name' if it exists, otherwise use a placeholder string with the identity
    source_name = record["n"]["name"] if "name" in record["n"] else f"Node_{source_id}"
    target_name = record["m"]["name"] if "name" in record["m"] else f"Node_{target_id}"
    
    # Add nodes and the edge to the graph
    G.add_node(source_id, label=source_name)
    G.add_node(target_id, label=target_name)
    G.add_edge(source_id, target_id)

# Close the Neo4j connection
neo4j_conn.close()

#Graph Diameter
try:
    diameter = nx.diameter(G)
except nx.NetworkXError as e:
    diameter = str(e)

print(f"Graph Diamter is:{diameter}")

#Network Resilience
G_copy = G.copy()
G_copy.remove_node(list(G.nodes())[0])
largest_component_size_after_removal = len(max(nx.connected_components(G_copy), key=len))

print(f"Largest component size after removal is: {largest_component_size_after_removal}")

#Query Performance
start_time = time.time()
shortest_paths = nx.shortest_path(G, source=0, target=10)
end_time = time.time()
query_time = end_time - start_time

print(f"Query time equal {query_time} seconds")

# # Define the filter function
def filter_results(data, threshold=0.0001):
     """ Filter results by a centrality value threshold """
     return {node_id: value for node_id, value in data.items() if value >= threshold}

# Calculate Degree Centrality
degree_centrality = nx.degree_centrality(G)
filtered_degree_centrality = filter_results(degree_centrality)
print("Filtered Degree Centrality:\n")
for node_id, value in filtered_degree_centrality.items():
    label = G.nodes[node_id]['label']
    print(f"{label} (ID {node_id}): {round(value, 4)}\n")


# Calculate and filter Betweenness Centrality
betweenness = nx.betweenness_centrality(G)
filtered_betweenness = filter_results(betweenness)
print("Filtered Betweenness Centrality:\n")
for node_id, value in filtered_betweenness.items():
    label = G.nodes[node_id]['label']
    print(f"{label} (ID {node_id}): {round(value, 4)}\n")

# Calculate and filter Closeness Centrality
closeness = nx.closeness_centrality(G)
filtered_closeness = filter_results(closeness)
print("Filtered Closeness Centrality:\n")
for node_id, value in filtered_closeness.items():
    label = G.nodes[node_id]['label']
    print(f"{label} (ID {node_id}): {round(value, 4)}\n")


def plot_centrality_measure(centrality_dict, title, ylabel):
    labels = [G.nodes[node_id]['label'] for node_id in centrality_dict]
    values = list(centrality_dict.values())

    # Sort the nodes by centrality values for better visualization
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    # Increase figure size for better readability if there are many nodes
    plt.figure(figsize=(20, 10))  # Adjust the width as necessary

    # Reduce bar width to increase space between bars
    bar_width = 0.5  # Less than 1 adds space between bars

    # Plot the bars with adjusted width
    bars = plt.bar(range(len(sorted_labels)), sorted_values, width=bar_width, color='skyblue', alpha=0.7)

    # Rotate labels for better visibility
    plt.xticks(ticks=range(len(sorted_labels)), labels=sorted_labels, rotation=90)

    # Adjust layout to make room for the labels
    plt.gcf().subplots_adjust(bottom=0.2)  # Increase the bottom margin

    # Add labels and title
    plt.xlabel('Nodes')
    plt.ylabel(ylabel)
    plt.title(title)

    # Display values on the bars (optional)
    for bar, value in zip(bars, sorted_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{round(value, 4)}", ha='center', va='bottom')

    plt.show()

# Plot each centrality measure
plot_centrality_measure(filtered_degree_centrality, "Filtered Degree Centrality", "Degree Centrality")
plot_centrality_measure(filtered_betweenness, "Filtered Betweenness Centrality", "Betweenness Centrality")
plot_centrality_measure(filtered_closeness, "Filtered Closeness Centrality", "Closeness Centrality")
# Calculate and filter Eigenvector Centrality
try:
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
except nx.PowerIterationFailedConvergence:
    eigenvector = nx.eigenvector_centrality_numpy(G)  # Fallback
filtered_eigenvector = filter_results(eigenvector)
print("Filtered Eigenvector Centrality:\n")
for node_id, value in filtered_eigenvector.items():
    label = G.nodes[node_id]['label']
    print(f"{label} (ID {node_id}): {round(value, 4)}\n")

# Apply filter to the eigenvector centrality if needed
filtered_eigenvector = {node_id: value for node_id, value in eigenvector.items() if value > 0.0001}  # Define your threshold

# Prepare data for plotting
labels = [G.nodes[node_id]['label'] for node_id in filtered_eigenvector.keys()]
values = list(filtered_eigenvector.values())

# Sort the nodes by centrality values for better visualization
sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
sorted_labels = [labels[i] for i in sorted_indices]
sorted_values = [values[i] for i in sorted_indices]

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_labels)), sorted_values, color='skyblue')

# Add labels to the x-ticks and rotate them for better visibility
plt.xticks(ticks=range(len(sorted_labels)), labels=sorted_labels, rotation=90)

# Add labels and title
plt.xlabel('Nodes')
plt.ylabel('Eigenvector Centrality')
plt.title('Filtered Eigenvector Centrality for Nodes')

# Display values on the bars (optional)
for i, value in enumerate(sorted_values):
    plt.text(i, value, f"{round(value, 4)}", ha='center', va='bottom')

plt.tight_layout()  # Adjust layout
plt.show()

# Community Detection
communities = list(greedy_modularity_communities(G))
print("Detected Communities:")
for i, community in enumerate(communities, 1):
    community_labels = [G.nodes[node]['label'] for node in community]
    print(f"Community {i}: {community_labels}")

pagerank = nx.pagerank(G)
filtered_pagerank = filter_results(pagerank)
for node, score in pagerank.items():
    label = G.nodes[node].get('label', f'Node {node}')
    print(f"{label} (ID {node}): {round(score, 4)}")

clustering = nx.clustering(G)
filtered_clustering = filter_results(clustering, threshold=0.0004)
for node, coeff in clustering.items():
    if coeff != 0:
        label = G.nodes[node].get('label', f'Node {node}')
        print(f"{label} (ID {node}): {round(coeff, 4)}")

def filter_path_lengths(data, threshold=5):
    """ Filter path lengths by a minimum path length threshold """
    filtered_data = {}
    for source, targets in data.items():
        # Filter targets based on the path length threshold
        filtered_targets = {target: length for target, length in targets.items() if length >= threshold}
        if filtered_targets:
            filtered_data[source] = filtered_targets
    return filtered_data

# Example of calculating and filtering path lengths
path_lengths = dict(nx.all_pairs_shortest_path_length(G))
filtered_paths = filter_path_lengths(path_lengths, threshold=2)

# Printing filtered path lengths
for source, targets in filtered_paths.items():
    source_label = G.nodes[source].get('label', f'Node {source}')
    for target, length in targets.items():
        target_label = G.nodes[target].get('label', f'Node {target}')
        print(f"Shortest path from {source_label} to {target_label} is {length} steps")

# Check if the graph is connected
# For directed graphs, use strongly_connected or weakly_connected
# Check if the directed graph is strongly connected
is_strongly_connected = nx.is_strongly_connected(G)
print(f"The graph is {'strongly connected' if is_strongly_connected else 'not strongly connected'}")

# Check if the directed graph is weakly connected
is_weakly_connected = nx.is_weakly_connected(G)
print(f"The graph is {'weakly connected' if is_weakly_connected else 'not weakly connected'}")

# Function to calculate common neighbor centrality for all non-adjacent node pairs
def common_neighbors_link_prediction(G):
    # Convert directed graph to undirected for common neighbor centrality
    g_undirected = G.to_undirected()
    return nx.common_neighbor_centrality(g_undirected)

# Get the common neighbors link prediction scores
cn_scores = common_neighbors_link_prediction(G)

# Get top scores and the nodes involved
top_cn_scores = sorted(cn_scores, key=lambda x: x[2], reverse=True)[:10]
print("Top Common Neighbors Scores:")
for u, v, score in top_cn_scores:
    u_label = G.nodes[u].get('label', f'Node {u}')
    v_label = G.nodes[v].get('label', f'Node {v}')
    print(f"({u_label}, {v_label}): {round(score, 4)}")

# # Using Kamada-Kawai layout for better spacing
# pos = nx.kamada_kawai_layout(G)
# for node, (x, y) in pos.items():
#     pos[node] = (x * 2, y * 2)  # Scale positions to increase space between nodes

# highlight_nodes = [u for u, v, score in top_cn_scores for node in (u, v)]
# highlight_nodes = list(set(highlight_nodes))

# # Plotly visualization setup
# edge_x = []
# edge_y = []
# for edge in G.edges():
#     x0, y0 = pos[edge[0]]
#     x1, y1 = pos[edge[1]]
#     edge_x.extend([x0, x1, None])
#     edge_y.extend([y0, y1, None])

# edge_trace = go.Scatter(
#     x=edge_x, y=edge_y,
#     line=dict(width=0.5, color='#888'),
#     hoverinfo='none',
#     mode='lines')

# node_x = []
# node_y = []
# node_text = []
# node_sizes = []

# # Define node size and color dynamically
# for node in G.nodes():
#     node_x.append(pos[node][0])
#     node_y.append(pos[node][1])
#     node_info = f"{G.nodes[node]['label']} (ID {node})"
#     node_text.append(node_info)
#     node_sizes.append(20 if node in highlight_nodes else 10)

# node_trace = go.Scatter(
#     x=node_x, y=node_y,
#     mode='markers+text',
#     text=node_text,
#     hoverinfo='text',
#     marker=dict(
#         showscale=False,
#         color=['red' if node in highlight_nodes else 'blue' for node in G.nodes()],
#         size=node_sizes,
#         line_width=2))

# # Create the figure
# fig = go.Figure(data=[edge_trace, node_trace],
#                 layout=go.Layout(
#                     showlegend=False,
#                     hovermode='closest',
#                     margin=dict(b=0, l=0, r=0, t=0),
#                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                     annotations=[dict(
#                         text="Node importance based on Common Neighbors",
#                         showarrow=False,
#                         xref="paper", yref="paper",
#                         x=0.005, y=-0.002 )],
#                     title="Network Graph with Highlighted Nodes"))

# fig.update_layout(title_font_size=16)
# fig.show()

# # Set node positions using a layout algorithm
# pos = nx.spring_layout(G)  # Tries to position nodes in a visually pleasing way

# # Draw the nodes, edges, and labels
# nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.6)
# nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
# nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()})

# plt.title('Network Graph')
# plt.axis('off')  # Turn off the axis
# plt.show()

from infomap import Infomap


# Assuming 'G' is your NetworkX graph
im = Infomap()

# Add edges to the Infomap object
for edge in G.edges():
    im.add_link(edge[0], edge[1])

# Run the Infomap clustering algorithm
im.run()

# Get community assignments
communities = im.get_modules()

# Now you can process or visualize the communities
print("Node to Community Assignments:", communities)

# Optionally, you can iterate through nodes in your graph and print their community
for node in G.nodes():
    community = communities[node]
    print(f"Node {node} is in community {community}")

