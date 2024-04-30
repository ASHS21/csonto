# This script is used to evaluate the CSONTO ontology against the UCO ontology and a domain dataset.
# It calculates various metrics such as precision, recall, coverage, cohesion, coupling, and semantic similarity.
# It also calculates structural quality metrics such as Instantiated Class Ratio (ICR), Instantiated Property Ratio (IPR),
# Subclass Property Acquisition (SPA), Inverse Multiple Inheritance (IMI), and Specific Property Inheritance (SPI).
# The script also extracts terms from a PDF document and CSV datasets to compare with the ontologies.
# Finally, it plots the results for comparison and visualization.

# Import necessary libraries
import pandas as pd
import fitz  # PyMuPDF
import rdflib
from rdflib import RDF, RDFS, OWL, Literal, URIRef
import re
from rdflib.namespace import OWL
from math import sqrt, pow
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import euclidean, cosine
from matplotlib_venn import venn2

# Define dictionaries to store class features
csonto_class_features = {}
uco_class_features = {}

# Load the ontology and extract terms
ontology_terms = set()
g = rdflib.Graph()
g.parse("/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/csonto-edit.rdf", format="application/rdf+xml")
for subj, pred, obj in g:
    if isinstance(obj, Literal):
        ontology_terms.add(str(obj).lower())

# Define the list of file paths
file_list = [
    "action/action.ttl", "analysis/analysis.ttl", "configuration/configuration.ttl", "core/core.ttl",
    "identity/identity.ttl", "location/location.ttl", "marking/marking.ttl", "master/uco.ttl",
    "observable/observable.ttl", "pattern/pattern.ttl", "role/role.ttl", "time/time.ttl", "tool/tool.ttl",
    "types/types.ttl", "victim/victim.ttl", "vocabulary/vocabulary.ttl"
]
# Load the new ontology
g_new = rdflib.Graph()

# Iterate over each file path and parse it
for file_path in file_list:
    result = g_new.parse(f"/Users/alialmoharif/Desktop/FYP/uco/{file_path}", format="turtle")

# Extract terms from the new ontology
new_ontology_terms = set()
for subj, pred, obj in g_new:
    if isinstance(obj, rdflib.Literal):
        new_ontology_terms.add(str(obj).lower())
    elif isinstance(subj, rdflib.URIRef):
        new_ontology_terms.add(str(subj).split('/')[-1].lower())  # Extracting a simplified local name

# Extract features for each class
for cls in g.subjects(RDF.type, OWL.Class):
    # Calculate subclass and property connections
    subclass_relations = set(g.objects(cls, RDFS.subClassOf))
    property_relations = set(g.predicates(cls, None))
    csonto_class_features[str(cls)] = {
        'properties': property_relations,
        'subclasses': subclass_relations
    }

# Extract features for each class
for cls in g.subjects(RDF.type, OWL.Class):
    # Calculate subclass and property connections
    subclass_relations = set(g_new.objects(cls, RDFS.subClassOf))
    property_relations = set(g_new.predicates(cls, None))
    uco_class_features[str(cls)] = {
        'properties': property_relations,
        'subclasses': subclass_relations
    }
# Additional metrics implementations
# Measure cohesion as the sum of property relations and subclass relations per class
def calculate_cohesion(g):
    cohesion_scores = {}
    for cls in g.subjects(RDF.type, OWL.Class):
        subclass_relations = set(g.objects(cls, RDFS.subClassOf))
        property_relations = set(g.predicates(cls, None))
        cohesion_scores[str(cls)] = len(subclass_relations) + len(property_relations)
    return cohesion_scores

# Measure coupling as the number of external class references per class
def calculate_coupling(g):
    coupling_counts = {}
    for cls in g.subjects(RDF.type, OWL.Class):
        external_refs = set()
        for _, _, obj in g.triples((cls, None, None)):
            if isinstance(obj, rdflib.URIRef) and not obj.startswith(str(g.namespace_manager.store.namespace(''))):
                external_refs.add(obj)
        coupling_counts[str(cls)] = len(external_refs)
    return coupling_counts

# Calculate the semantic similarity between classes
def calculate_semantic_similarity(g, class_features):
    similarity_scores = {}
    # Assuming feature extraction has been done and is available in `class_features`
    for i, cls1 in enumerate(class_features):
        for j, cls2 in enumerate(class_features):
            if i < j:  # To avoid calculating the same pair twice
                prop_intersection = len(class_features[cls1]['properties'].intersection(class_features[cls2]['properties']))
                prop_union = len(class_features[cls1]['properties'].union(class_features[cls2]['properties']))
                if prop_union > 0:
                    similarity = prop_intersection / prop_union
                else:
                    similarity = 0
                similarity_scores[(cls1, cls2)] = similarity
    return similarity_scores

# Calculate the average semantic similarity between classes
def calculate_average_semantic_similarity(g, class_features):
    similarity_scores = {}
    total_similarity = 0
    count = 0
    for i, cls1 in enumerate(class_features):
        for j, cls2 in enumerate(class_features):
            if i < j:
                prop_intersection = len(class_features[cls1]['properties'].intersection(class_features[cls2]['properties']))
                prop_union = len(class_features[cls1]['properties'].union(class_features[cls2]['properties']))
                similarity = prop_intersection / prop_union if prop_union > 0 else 0
                total_similarity += similarity
                count += 1
                similarity_scores[(cls1, cls2)] = similarity
    average_similarity = total_similarity / count if count > 0 else 0
    return average_similarity

# Function to clean and extract terms from a text block
def extract_terms(text_block):
    return re.findall(r'\b\w+\b', text_block.lower())  # Extract words and convert to lower case

# Calculate the total number of instances in the ontology
def calculate_total_instances(graph):
    class_instances = set(graph.subjects(RDF.type, OWL.Class))
    instance_count = sum(len(set(graph.subjects(RDF.type, cls))) for cls in class_instances)
    return instance_count

total_instances = calculate_total_instances(g)

# Calculate the total number of instances in the ontology
def calculate_total_instances(graph):
    instances = set()
    for cls in graph.subjects(RDF.type, OWL.Class):
        instances.update(graph.subjects(RDF.type, cls))
    return len(instances)

total_instances = calculate_total_instances(g)

# Helper function to calculate depth of a class
def calculate_depth(cls, graph, depth=0, visited=None):
    if visited is None:
        visited = set()
    if cls in visited:
        return depth
    visited.add(cls)
    depths = [calculate_depth(parent, graph, depth + 1, visited) for parent in graph.objects(cls, RDFS.subClassOf)]
    return max(depths, default=depth)

# Calculate CI for a specific class
def calculate_ci(cls, graph, total_instances, depth=0):
    ir = len(set(graph.subjects(RDF.type, cls))) / total_instances
    ci = ir / (2 ** depth)
    for subclass in graph.subjects(RDFS.subClassOf, cls):
        if subclass != cls:  # Prevent recursion on the same class
            ci += calculate_ci(subclass, graph, total_instances, depth + 1)
    return ci

# Calculate the Specific Property Inheritance (SPI) for a class
def get_subclass_specific_properties(cls, graph):
    properties = set(graph.predicates(subject=cls))
    for superclass in graph.objects(cls, RDFS.subClassOf):
        properties -= set(graph.predicates(subject=superclass))
    return properties

# Metric calculation functions
def calculate_icr(graph):
    total_classes = set(graph.subjects(RDF.type, OWL.Class))
    instantiated_classes = {cls for cls in total_classes if list(graph.subjects(RDF.type, cls))}
    return len(instantiated_classes) / len(total_classes)

# Calculate the Instantiated Property Ratio (IPR)
def calculate_ipr(graph):
    total_properties = set(graph.predicates())
    instantiated_properties = {prop for prop in total_properties if list(graph.objects(predicate=prop))}
    return len(instantiated_properties) / len(total_properties)

# Calculate the Subclass Property Acquisition (SPA)
def calculate_spa(graph):
    total_classes = set(graph.subjects(RDF.type, OWL.Class))
    spa_sum = sum(len(get_subclass_specific_properties(cls, graph)) for cls in total_classes)
    return spa_sum / len(total_classes)

# Calculate the Inverse Multiple Inheritance (IMI)
def calculate_imi(graph):
    total_classes = set(graph.subjects(RDF.type, OWL.Class))
    imi_sum = sum(1 / len(set(graph.objects(cls, RDFS.subClassOf))) for cls in total_classes if list(graph.objects(cls, RDFS.subClassOf)))
    return imi_sum / len(total_classes)

# Calculate the Specific Property Inheritance (SPI) for a class
def calculate_spi_for_class(cls, graph):
    specific_properties = get_subclass_specific_properties(cls, graph)
    used_properties = {prop for prop in specific_properties if list(graph.objects(predicate=prop))}
    total_triples = len(list(graph.triples((None, None, cls))))
    return len(used_properties) / total_triples if total_triples else 0

# Calculate the average Specific Property Inheritance (SPI) for all classes
def calculate_spi_per_class(graph):
    return {cls: calculate_spi_for_class(cls, graph) for cls in graph.subjects(RDF.type, OWL.Class)}

# Calculate the average Specific Property Inheritance (SPI) for all classes
def calculate_spi_average(graph):
    spi_values = calculate_spi_per_class(graph)
    return sum(spi_values.values()) / len(spi_values) if spi_values else 0

# Extract terms from the PDF
with fitz.open("/Users/alialmoharif/Desktop/FYP/Dataset/Cybersecurity Acronyms 2020.pdf") as pdf_document:
    all_text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        all_text += page.get_text()
terms_from_pdf = set(extract_terms(all_text))

# Calculate cohesion and coupling
cohesion_scores = calculate_cohesion(g)
coupling_counts = calculate_coupling(g)
semantic_similarity_scores = calculate_semantic_similarity(g, csonto_class_features)

#Calculate the cohesion and coupling and semantic similarity scores for UCO
cohesion_scores2 = calculate_cohesion(g_new)
coupling_counts2 = calculate_coupling(g_new)
semantic_similarity_scores2 = calculate_semantic_similarity(g_new, uco_class_features)

# # Display cohesion and coupling for each class
# for cls in cohesion_scores2:
#     if cohesion_scores2[cls] > 0:
#         print(f"Cohesion for {cls}: {cohesion_scores2[cls]}")
#         if coupling_counts2.get(cls, 0) > 0:
#             print(f"Coupling for {cls}: {coupling_counts2.get(cls, 0)}")

# # Display semantic similarity scores (limited output for brevity)
# print("Semantic Similarity Scores:")
# for pair, score in list(semantic_similarity_scores2.items()):  
#     if score > 0:
#         print(f"{pair}: {score}")

# # Display cohesion and coupling for each class
# for cls in cohesion_scores:
#     if cohesion_scores[cls] > 0:
#          print(f"Cohesion for {cls}: {cohesion_scores[cls]}")
#          if coupling_counts.get(cls, 0) > 0:
#              print(f"Coupling for {cls}: {coupling_counts.get(cls, 0)}")

# # Display semantic similarity scores (limited output for brevity)
# print("Semantic Similarity Scores:")
# for pair, score in list(semantic_similarity_scores.items()):  
#     if score > 0:
#         print(f"{pair}: {score}")


# # Perform calculations for structual Quality metrics
# spi_values2 = calculate_spi_per_class(g_new)
# icr2 = calculate_icr(g_new)
# ipr2 = calculate_ipr(g_new)
# spa2 = calculate_spa(g_new)
# imi2 = calculate_imi(g_new)

# print("\nSPI for each class in UCO:")
# for cls, spi in spi_values2.items():
#     if spi > 0:
#         print(f"{cls}: {spi}")

# print(f"\nInstantiated Class Ratio (ICR): {icr2}")
# print(f"Instantiated Property Ratio (IPR): {ipr2}")
# print(f"Subclass Property Acquisition (SPA): {spa2}")
# print(f"Inverse Multiple Inheritance (IMI): {imi2}")


# Load CSV datasets and extract terms
csv_files = ['/Users/alialmoharif/Desktop/FYP/Dataset/5.12 Cybersecurity Detail.csv',
             '/Users/alialmoharif/Desktop/FYP/Dataset/Access Log Jan 01 2017.csv',
             '/Users/alialmoharif/Desktop/FYP/Dataset/Cybersecurity Summary.csv']

# Extract terms from each column in the CSV files
dataset_terms = set()
for file in csv_files:
    df = pd.read_csv(file)
    for column in df.columns:
        dataset_terms.update(extract_terms(" ".join(df[column].astype(str))))

# Combine all terms from PDF and CSV datasets
combined_terms = terms_from_pdf.union(dataset_terms)

# Comparison between the CSONTO and the domain dataset terms
# Match ontology terms with combined terms to find true positives
true_positives1 = ontology_terms.intersection(combined_terms)

# Calculate metrics
predicted_positives1 = len(ontology_terms)  # All ontology terms considered as predictions
actual_positives1 = len(combined_terms)  # All terms from domain considered as actual positives

# Calculate new precision, recall, and coverage
precision1 = len(true_positives1) / predicted_positives1 if predicted_positives1 else 0
recall1 = len(true_positives1) / actual_positives1 if actual_positives1 else 0
coverage1 = len(true_positives1) / actual_positives1 if actual_positives1 else 0

# Output results
print(f"Total CSONTO Terms: {predicted_positives1}")
print(f"Total Combined Terms from Domain: {actual_positives1}")
print(f"Relevant CSONTO Terms Identified: {len(true_positives1)}")
print(f"True Positives: {len(true_positives1)}")
print(f"False Positives: {predicted_positives1 - len(true_positives1)}")
print(f"False Negatives: {actual_positives1 - len(true_positives1)}")
print(f"Coverage: {coverage1:.2f}")
print(f"Precision: {precision1:.2f}")
print(f"Recall: {recall1:.2f}")

print("\n")

# Perform calculations for Structrual Quality metrics
spi_values = calculate_spi_per_class(g)
icr = calculate_icr(g)
ipr = calculate_ipr(g)
spa = calculate_spa(g)
imi = calculate_imi(g)
# #Calculate and print CI for all classes
# print("CI for each class:")
# for cls in g.subjects(RDF.type, OWL.Class):
#     cls_depth = calculate_depth(cls, g)
#     cls_ci = calculate_ci(cls, g, total_instances, cls_depth)
#     print(f"{cls}: {cls_ci}")

# print("\nSPI for each class in CSONTO:")
# for cls, spi in spi_values.items():
#     if spi > 0:
#         print(f"{cls}: {spi}")

# print(f"\nInstantiated Class Ratio (ICR): {icr}")
# print(f"Instantiated Property Ratio (IPR): {ipr}")
# print(f"Subclass Property Acquisition (SPA): {spa}")
# print(f"Inverse Multiple Inheritance (IMI): {imi}")


# Compare the UCO with combined domain terms
true_positives2 = new_ontology_terms.intersection(combined_terms)
predicted_positives2 = len(new_ontology_terms)
actual_positives2 = len(combined_terms)

# Calculate new precision, recall, and coverage
precision2 = len(true_positives2) / predicted_positives2 if predicted_positives2 else 0
recall2 = len(true_positives2) / actual_positives2 if actual_positives2 else 0
coverage2 = len(true_positives2) / actual_positives2 if actual_positives2 else 0

# Print results
print(f"Total UCO Terms: {predicted_positives2}")
print(f"True Positives: {len(true_positives2)}")
print(f"False Positives: {predicted_positives2 - len(true_positives2)}")
print(f"False Negatives: {actual_positives2 - len(true_positives2)}")
print(f"Coverage: {coverage2:.2f}")
print(f"Precision: {precision2:.2f}")
print(f"Recall: {recall2:.2f}")


# Comparison between CSONTO and UCO
tp_ontologies_terms = ontology_terms.intersection(new_ontology_terms)
predicted_positives3 = len(ontology_terms)
actual_positives3 = len(new_ontology_terms)
precision3 = len(tp_ontologies_terms) / predicted_positives3 if predicted_positives3 else 0
recall3 = len(tp_ontologies_terms) / actual_positives3 if actual_positives3 else 0
coverage3 = len(tp_ontologies_terms) / actual_positives3 if actual_positives3 else 0

print(f"Total CSONTO Terms: {predicted_positives3}")
print(f"Total UCO Terms: {actual_positives3}")
print(f"True Positives: {len(tp_ontologies_terms)}")
print(f"False Positives: {predicted_positives3 - len(tp_ontologies_terms)}")
print(f"False Negatives: {actual_positives3 - len(tp_ontologies_terms)}")
print(f"Coverage: {coverage3:.5f}")
print(f"Precision: {precision3:.5f}")
print(f"Recall: {recall3:.5f}")

plt.style.use('ggplot')
# Metrics names and their values for each ontology
metrics = ['Total Ontology Terms', 'True Positives identified', 'False Positives', 'False Negatives', 'Coverage', 'Precision', 'Recall']
values_original = [1802, 22, 1780, 2933, 0.01, 0.01, 0.01]
values_new = [3242, 136, 3106, 2819, 0.05, 0.04, 0.05]

# Define the width of the bars
width = 0.35

# Plotting each metric separately
for i, metric in enumerate(metrics):
    fig, ax = plt.subplots(figsize=(8, 6))  # Size of individual plot
    x = np.arange(1)  # Only one group per metric
    rects1 = ax.bar(x - width/2, values_original[i], width, label='CSONTO', color='b', alpha=0.6)
    rects2 = ax.bar(x + width/2, values_new[i], width, label='UCO', color='r', alpha=0.6)
    
    # Setting chart title and labels
    ax.set_ylabel('Values')
    ax.set_title(f'{metric}')
    ax.set_xticks([])
    ax.legend()

    # Adding value labels on top of bars
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_value_labels(rects1)
    add_value_labels(rects2)
    
    plt.show()



# Define the metrics and their values for each ontology
def plot_structural_metrics(g_csonto, g_uco):
    # Calculate metrics for CSONTO
    csonto_icr = calculate_icr(g_csonto)
    csonto_ipr = calculate_ipr(g_csonto)
    csonto_spa = calculate_spa(g_csonto)
    csonto_imi = calculate_imi(g_csonto)
    csonto_spi = calculate_spi_average(g_csonto)
    
    # Calculate metrics for UCO
    uco_icr = calculate_icr(g_uco)
    uco_ipr = calculate_ipr(g_uco)
    uco_spa = calculate_spa(g_uco)
    uco_imi = calculate_imi(g_uco)
    uco_spi = calculate_spi_average(g_uco)
    
    # Data for plotting
    metrics = ['ICR', 'IPR', 'SPA', 'IMI', 'Average SPI']
    csonto_values = [csonto_icr, csonto_ipr, csonto_spa, csonto_imi, csonto_spi]
    uco_values = [uco_icr, uco_ipr, uco_spa, uco_imi, uco_spi]
    
    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, csonto_values, width, label='CSONTO', color='blue')
    rects2 = ax.bar(x + width/2, uco_values, width, label='UCO', color='red')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Structural Quality Metrics per Ontology')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Adding value labels on top of bars
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_value_labels(rects1)
    add_value_labels(rects2)

    plt.show()

# Call the function with the graphs loaded with CSONTO and UCO data
plot_structural_metrics(g, g_new) 

csonto_semantic_similarity = calculate_average_semantic_similarity(g, csonto_class_features)
uco_semantic_similarity = calculate_average_semantic_similarity(g_new, uco_class_features)

# Improved aesthetics with ggplot style
plt.style.use('ggplot')

# Define a function for better value labeling
def add_value_labels(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Define the general function for plotting comparisons
def plot_comparison(metrics, values_ontology1, values_ontology2, ontology1_label, ontology2_label, title):
    x = np.arange(len(metrics))  # Label locations
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 7))  # Figure size for readability
    rects1 = ax.bar(x - width/2, values_ontology1, width, label=ontology1_label, color='#307EC7', alpha=0.8)
    rects2 = ax.bar(x + width/2, values_ontology2, width, label=ontology2_label, color='#E1812C', alpha=0.8)

    # Chart title and labels
    ax.set_ylabel('Values', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(frameon=False, fontsize=12)

    # Value labels on top of bars
    add_value_labels(ax, rects1)
    add_value_labels(ax, rects2)

    # Adding gridlines
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

    # Removing the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()  # Adjust layout
    plt.show()

# Define the metrics and their values for each ontology
csonto_metrics = ['Cohesion', 'Coupling', 'Semantic Similarity']
csonto_values = [
    sum(calculate_cohesion(g).values()) / len(calculate_cohesion(g)),
    sum(calculate_coupling(g).values()) / len(calculate_coupling(g)),
    calculate_average_semantic_similarity(g, csonto_class_features)
]

# Call the function to plot the comparison
uco_metrics = ['Cohesion', 'Coupling', 'Semantic Similarity']
uco_values = [
    sum(calculate_cohesion(g_new).values()) / len(calculate_cohesion(g_new)),
    sum(calculate_coupling(g_new).values()) / len(calculate_coupling(g_new)),
    calculate_average_semantic_similarity(g_new, uco_class_features)
]

# Call the function to plot the comparison
plot_comparison(csonto_metrics, csonto_values, uco_values, 'CSONTO', 'UCO', 'Structural Quality Metrics Comparison')

# Define the metrics and their values for each ontology and plot them as venn diagram
venn2([new_ontology_terms, combined_terms], ('UCO', 'Dataset'))
plt.show()



