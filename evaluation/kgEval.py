import json
from neo4j import GraphDatabase
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms import link_prediction
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import plotly.graph_objects as go
import py2neo
import community as community_louvain
import scipy 
import re
import fitz 
import pandas as pd
import time
from infomap import Infomap
import sys

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

    def fetch_graph_data(self, query):
        with self.driver.session() as session:
            results = session.run(query)
            records = list(results)  
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

output_file_path = 'analysis_results.txt'

# Open the output file in write mode and redirect stdout to this file
with open(output_file_path, 'w') as output_file:
    original_stdout = sys.stdout  
    sys.stdout = output_file  

    neo4j_conn = Neo4jConnection("/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/pages/config.json")

    

    def generate_latex_documents(G):
                    sections = {
                        'Detected Communities': generate_communities_section,
                        'PageRank': generate_pagerank_section,
                        'Clustering Coefficient': generate_clustering_coefficient_section,
                        'Path Lengths': generate_path_lengths_section,
                        'Connectivity': generate_connectivity_section,
                        'Common Neighbors Scores': generate_common_neighbors_scores_section,
                        # 'Community Assignments': generate_community_assignments_section,  
                    }

                    for title, func in sections.items():
                        filename = title.lower().replace(' ', '_') + '.tex'
                        with open(filename, 'w') as file:
                            file.write("\\documentclass{article}\n")
                            file.write("\\usepackage{longtable}\n")
                            file.write("\\begin{document}\n")
                            file.write(f"\\section*{{{title}}}\n")
                            func(G, file)
                            file.write("\\end{document}\n")

    def generate_communities_section(G, file):
        communities = list(greedy_modularity_communities(G))
        for i, community in enumerate(communities, 1):
            community_labels = [G.nodes[node]['label'] for node in community]
            file.write(f"Community {i}: " + ", ".join(community_labels) + "\n\n")

    def generate_pagerank_section(G, file):
        pagerank = nx.pagerank(G)
        file.write("\\begin{longtable}{ll}\n")
        for node, score in pagerank.items():
            label = G.nodes[node].get('label', f'Node {node}')
            file.write(f"{label} & {score:.4f} \\\\\n")
        file.write("\\end{longtable}\n\n")

    def generate_clustering_coefficient_section(G, file):
        clustering = nx.clustering(G)
        file.write("\\begin{longtable}{ll}\n")
        for node, coeff in clustering.items():
            if coeff != 0:
                label = G.nodes[node].get('label', f'Node {node}')
                file.write(f"{label} & {coeff:.4f} \\\\\n")
        file.write("\\end{longtable}\n\n")

    def generate_path_lengths_section(G, file):
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        for source, targets in path_lengths.items():
            source_label = G.nodes[source].get('label', f'Node {source}')
            for target, length in targets.items():
                target_label = G.nodes[target].get('label', f'Node {target}')
                file.write(f"Shortest path from {source_label} to {target_label} is {length} steps\n")

    def generate_connectivity_section(G, file):
        is_strongly_connected = nx.is_strongly_connected(G)
        file.write(f"The graph is {'strongly' if is_strongly_connected else 'not strongly'} connected.\n")
        is_weakly_connected = nx.is_weakly_connected(G)
        file.write(f"The graph is {'weakly' if is_weakly_connected else 'not weakly'} connected.\n\n")

    def generate_common_neighbors_scores_section(G, file):
        
        cn_scores = common_neighbors_link_prediction(G)
        for u, v, score in cn_scores:
            u_label = G.nodes[u].get('label', f'Node {u}')
            v_label = G.nodes[v].get('label', f'Node {v}')
            file.write(f"({u_label}, {v_label}): {score:.4f} \\\\\n")

        
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
                json_data = json.load(file)  
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

    # Path to JSON files
    json_file_path = "/Users/alialmoharif/Desktop/D3fend Full Mappings.json"
    csv_file_path = '/Users/alialmoharif/Desktop/CSKG-FYP.csv'
    pdf_path = "/Users/alialmoharif/Desktop/FYP/Dataset/Cybersecurity Acronyms 2020.pdf"
    csv_files = [
        "/Users/alialmoharif/Desktop/FYP/Dataset/5.12 Cybersecurity Detail.csv",
        "/Users/alialmoharif/Desktop/FYP/Dataset/Access Log Jan 01 2017.csv",
        "/Users/alialmoharif/Desktop/FYP/Dataset/Cybersecurity Summary.csv"
    ]

    # Extract terms from datasets
    # Extract terms from the gold standard JSON
    defend_terms = load_and_extract_terms_from_json(json_file_path)
    terms_from_pdf = extract_terms_from_pdf(pdf_path)
    terms_from_csvs = extract_terms_from_csvs(csv_files)
    all_dataset_terms = terms_from_pdf.union(terms_from_csvs)


    kg_terms = extract_terms_from_csv(csv_file_path)  

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

    print("Evaluation between the KG and the domain datasets")
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

    N, E = neo4j_conn.fetch_graph_data2()

    # Calculate sparsity beta
    beta = 1 - (E / (N * (N - 1)))  # for directed graph

    # Print out the KG properties and the calculated metrics
    print(f"Number of nodes (N): {N}")
    print(f"Number of edges (E): {E}")
    print(f"Sparsity (beta): {beta}")

    queries = {
            "reports_to and part_of": "MATCH (n:CyberSecurityScore)<-[r:REPORTS_TO]-(m) RETURN n, r, m UNION MATCH (n)<-[r:PART_OF]-(m) RETURN n, r, m",
            "reports_to": "MATCH (n:CyberSecurityScore)<-[r:REPORTS_TO]-(m) RETURN n, r, m ",
            "define_policy" : "MATCH(n:Governance)-[r:DEFINE_POLICY]->(m) RETURN n,r,m",
            "full_graph": "MATCH (n)-[r]-(m) RETURN n, r, m"

        }

    for query_name, query in queries.items():
            records = neo4j_conn.fetch_graph_data(query)
            if not records:
                print(f"No records to process for {query_name}.")
                continue

            G = nx.DiGraph()

            # Populate the graph with nodes and edges along with labels
            for record in records:
                n, m = record['n'], record['m']
                source_id, target_id = n.id, m.id
                source_name = n.get('name', f'Node_{source_id}')
                target_name = m.get('name', f'Node_{target_id}')

                G.add_node(source_id, label=source_name)
                G.add_node(target_id, label=target_name)
                G.add_edge(source_id, target_id)

            #Graph Diameter
            try:
                diameter = nx.diameter(G)
            except nx.NetworkXError as e:
                diameter = str(e)

            print(f"Graph Diamter is:{diameter}")

            #Network Resilience for Directed Graph
            G_copy = G.copy()
            G_copy.remove_node(list(G.nodes())[0])

            # Use strongly_connected_components or weakly_connected_components 
            if nx.is_strongly_connected(G_copy):
                largest_component_size_after_removal = len(max(nx.strongly_connected_components(G_copy), key=len))
            else:
                largest_component_size_after_removal = len(max(nx.weakly_connected_components(G_copy), key=len))

            print(f"Largest component size after removal is: {largest_component_size_after_removal}")

            # Query Performance
            start_time = time.time()

            # Check if both source and target nodes exist in the graph
            source_node = 0
            target_node = 10

            if G.has_node(source_node) and G.has_node(target_node):
                shortest_paths = nx.shortest_path(G, source=source_node, target=target_node)
                print(f"Shortest path from {source_node} to {target_node}: {shortest_paths}")
            else:
                if not G.has_node(source_node):
                    print(f"Source node {source_node} not found in graph.")
                if not G.has_node(target_node):
                    print(f"Target node {target_node} not found in graph.")

            end_time = time.time()
            query_time = end_time - start_time

            print(f"Query time equal {query_time} seconds")

            # Calculate metrics
            density = nx.density(G)
            #avg_path_length = nx.average_shortest_path_length(G)
            #diameter = nx.diameter(G)
            global_clustering_coefficient = nx.transitivity(G)
            local_clustering_coefficients = nx.average_clustering(G)
            #connected_components = nx.number_connected_components(G)

            print("Density:", density)
            #print("Average Path Length:", avg_path_length)
            print("Diameter:", diameter)
            print("Global Clustering Coefficient:", global_clustering_coefficient)
            print("Average Local Clustering Coefficient:", local_clustering_coefficients)
            #print("Connected Components:", connected_components)
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
            filtered_eigenvector = {node_id: value for node_id, value in eigenvector.items() if value > 0.0001}  

            def plot_centrality_measure(G, centrality_dict, title, ylabel):
                # Exclude zero values
                non_zero_centrality = {node: value for node, value in centrality_dict.items() if value > 0}
                
                if not non_zero_centrality:
                    print(f"No non-zero centrality values to plot for {title}.")
                    return

                # If the graph is large, focus on the top nodes only
                if len(G) > 30:
                    print(f"Graph is too large to plot {title} for all nodes. Showing top nodes only.")
                    top_nodes = sorted(non_zero_centrality, key=non_zero_centrality.get, reverse=True)[:30]
                    top_centrality = {node: non_zero_centrality[node] for node in top_nodes}
                else:
                    top_centrality = non_zero_centrality

                labels = [G.nodes[node_id]['label'] for node_id in top_centrality.keys()]
                values = list(top_centrality.values())
                sorted_values, sorted_labels = zip(*sorted(zip(values, labels), reverse=True))

                # Set up figure and axes
                fig, ax = plt.subplots(figsize=(10, 6))
                bar_color = 'steelblue'

                # Create bar plot
                bars = ax.bar(range(len(sorted_labels)), sorted_values, color=bar_color)

                # Customize ticks and labels
                ax.set_xticks(range(len(sorted_labels)))
                ax.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=10)
                ax.set_xlabel('Nodes', fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(title, fontsize=14)

                # Add gridlines
                ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

                # Limit the number of decimal places for bar annotations
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

                plt.tight_layout()
                plt.show()


            # Plot each centrality measures
            plot_centrality_measure(G,filtered_degree_centrality, "Filtered Degree Centrality", "Degree Centrality")
            plot_centrality_measure(G,filtered_betweenness, "Filtered Betweenness Centrality", "Betweenness Centrality")
            plot_centrality_measure(G,filtered_closeness, "Filtered Closeness Centrality", "Closeness Centrality")
            
            plot_centrality_measure(G, filtered_eigenvector, "Filtered Eigenvector Centrality", "Eigenvector Centrality")

            # # Latex Table for Centrality Measures
            # generate_latex_for_centrality(degree_centrality, filename='centrality_measures.tex')
            # generate_latex_for_centrality(betweenness, filename='betweenness_measures.tex')
            # generate_latex_for_centrality(closeness, filename='closeness_measures.tex')
            # generate_latex_for_centrality(eigenvector, filename='eigenvector_measures.tex')

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

            # Assuming 'G' is your NetworkX graph
            im = Infomap()

            # Add edges to the Infomap object
            for edge in G.edges():
                im.add_link(edge[0], edge[1])

            # Run the Infomap clustering algorithm
            im.run()
            
            # Print the community assignments
            print("Node to Community Assignments:")
            communities = im.get_modules()
            for node, module in communities.items():
                print(f"Node {node} is in community {module}")

            # Now you can process or visualize the communities
            print("Node to Community Assignments:", communities)

            # Optionally, you can iterate through nodes in your graph and print their community
            for node in G.nodes():
                community = communities[node]
                print(f"Node {node} is in community {community}")

            generate_latex_documents(G)

            # Close the Neo4j connection
            neo4j_conn.close()

        
            print(f"Processed graph for query '{query_name}' with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                    


    # Reset the standard output to its original value
    sys.stdout = original_stdout

# After this point, any print statement will output to the terminal as usual
print("Done with all processing, results saved to file.")