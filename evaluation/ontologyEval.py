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

class_features = {}

# Load the ontology and extract terms
ontology_terms = set()
g = rdflib.Graph()
g.parse("/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/csonto-edit.rdf", format="application/rdf+xml")
for subj, pred, obj in g:
    if isinstance(obj, Literal):
        ontology_terms.add(str(obj).lower())

# Extract features for each class
for cls in g.subjects(RDF.type, OWL.Class):
    # Calculate subclass and property connections
    subclass_relations = set(g.objects(cls, RDFS.subClassOf))
    property_relations = set(g.predicates(cls, None))
    class_features[str(cls)] = {
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

def calculate_semantic_similarity(g):
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

# Calculate cohesion and coupling
cohesion_scores = calculate_cohesion(g)
coupling_counts = calculate_coupling(g)
semantic_similarity_scores = calculate_semantic_similarity(g)

# Display cohesion and coupling for each class
for cls in cohesion_scores:
    print(f"Cohesion for {cls}: {cohesion_scores[cls]}")
    print(f"Coupling for {cls}: {coupling_counts.get(cls, 0)}")

# Display semantic similarity scores (limited output for brevity)
print("Semantic Similarity Scores:")
for pair, score in list(semantic_similarity_scores.items())[:10]:  # Displaying the first 10 for brevity
    print(f"{pair}: {score}")


# Function to clean and extract terms from a text block
def extract_terms(text_block):
    return re.findall(r'\b\w+\b', text_block.lower())  # Extract words and convert to lower case

# Extract terms from the PDF
with fitz.open("/Users/alialmoharif/Desktop/FYP/Dataset/Cybersecurity Acronyms 2020.pdf") as pdf_document:
    all_text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        all_text += page.get_text()
terms_from_pdf = set(extract_terms(all_text))

# Load CSV datasets and extract terms
csv_files = ['/Users/alialmoharif/Desktop/FYP/Dataset/5.12 Cybersecurity Detail.csv',
             '/Users/alialmoharif/Desktop/FYP/Dataset/Access Log Jan 01 2017.csv',
             '/Users/alialmoharif/Desktop/FYP/Dataset/Cybersecurity Summary.csv']
dataset_terms = set()
for file in csv_files:
    df = pd.read_csv(file)
    for column in df.columns:
        dataset_terms.update(extract_terms(" ".join(df[column].astype(str))))

# Combine all terms from PDF and CSV datasets
combined_terms = terms_from_pdf.union(dataset_terms)

# Match ontology terms with combined terms to find true positives
true_positives = ontology_terms.intersection(combined_terms)

# Calculate metrics
predicted_positives = len(ontology_terms)  # All ontology terms considered as predictions
actual_positives = len(combined_terms)  # All terms from domain considered as actual positives

precision = len(true_positives) / predicted_positives if predicted_positives else 0
recall = len(true_positives) / actual_positives if actual_positives else 0
coverage = len(true_positives) / actual_positives if actual_positives else 0

# Output results
print(f"Total CSONTO Terms: {predicted_positives}")
print(f"Total Combined Terms from Domain: {actual_positives}")
print(f"Relevant CSONTO Terms Identified: {len(true_positives)}")
print(f"True Positives: {len(true_positives)}")
print(f"False Positives: {predicted_positives - len(true_positives)}")
print(f"False Negatives: {actual_positives - len(true_positives)}")
print(f"Coverage: {coverage:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

print("\n")


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

# Calculate and print CI for all classes
print("CI for each class:")
for cls in g.subjects(RDF.type, OWL.Class):
    cls_depth = calculate_depth(cls, g)
    cls_ci = calculate_ci(cls, g, total_instances, cls_depth)
    print(f"{cls}: {cls_ci}")

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

def calculate_ipr(graph):
    total_properties = set(graph.predicates())
    instantiated_properties = {prop for prop in total_properties if list(graph.objects(predicate=prop))}
    return len(instantiated_properties) / len(total_properties)


def calculate_spa(graph):
    total_classes = set(graph.subjects(RDF.type, OWL.Class))
    spa_sum = sum(len(get_subclass_specific_properties(cls, graph)) for cls in total_classes)
    return spa_sum / len(total_classes)

def calculate_imi(graph):
    total_classes = set(graph.subjects(RDF.type, OWL.Class))
    imi_sum = sum(1 / len(set(graph.objects(cls, RDFS.subClassOf))) for cls in total_classes if list(graph.objects(cls, RDFS.subClassOf)))
    return imi_sum / len(total_classes)

def calculate_spi_for_class(cls, graph):
    specific_properties = get_subclass_specific_properties(cls, graph)
    used_properties = {prop for prop in specific_properties if list(graph.objects(predicate=prop))}
    total_triples = len(list(graph.triples((None, None, cls))))
    return len(used_properties) / total_triples if total_triples else 0

def calculate_spi_per_class(graph):
    return {cls: calculate_spi_for_class(cls, graph) for cls in graph.subjects(RDF.type, OWL.Class)}

# Perform calculations
spi_values = calculate_spi_per_class(g)
icr = calculate_icr(g)
ipr = calculate_ipr(g)
spa = calculate_spa(g)
imi = calculate_imi(g)

print("\nSPI for each class:")
for cls, spi in spi_values.items():
    print(f"{cls}: {spi}")

print(f"\nInstantiated Class Ratio (ICR): {icr}")
print(f"Instantiated Property Ratio (IPR): {ipr}")
print(f"Subclass Property Acquisition (SPA): {spa}")
print(f"Inverse Multiple Inheritance (IMI): {imi}")
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

# Compare with combined domain terms
true_positives = new_ontology_terms.intersection(combined_terms)
predicted_positives = len(new_ontology_terms)
actual_positives = len(combined_terms)

# Calculate new precision, recall, and coverage
precision = len(true_positives) / predicted_positives if predicted_positives else 0
recall = len(true_positives) / actual_positives if actual_positives else 0
coverage = len(true_positives) / actual_positives if actual_positives else 0

# Print results
print(f"Total UCO Terms: {predicted_positives}")
print(f"True Positives: {len(true_positives)}")
print(f"False Positives: {predicted_positives - len(true_positives)}")
print(f"False Negatives: {actual_positives - len(true_positives)}")
print(f"Coverage: {coverage:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


# Metrics names and their values for each ontology
metrics = ['Total Ontology Terms', 'True Positives', 'False Positives', 'False Negatives', 'Coverage', 'Precision', 'Recall']
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