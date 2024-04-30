# This script is used to evaluate the ontology model by calculating various metrics and visualizing the results.
# It uses the Owlready2 library to load the ontology and calculate structural metrics such as class count, property count, and individual count.
# The script also includes functions to calculate hierarchical metrics, complexity metrics, and ontology richness metrics.
# It uses a reasoner to check consistency and normalize subclass and equivalent axioms.
# The calculated metrics are then visualized using bar charts and other plots to provide insights into the ontology's structure and complexity.
# The script also includes code to analyze subclass distribution and property specificity, which can help identify areas of improvement in the ontology design.
# The final part of the script demonstrates how to plot the subclass distribution and property specificity using bar charts and print the properties with the highest domain and range specificity.





# Import necessary libraries
from owlready2 import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set the style for the plots
sns.set(style="whitegrid")

# Load the ontology and run a reasoner
onto = get_ontology("/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/csonto-edit.rdf").load()
sync_reasoner_pellet()

# Print the classes and their superclasses
for cls in onto.classes():
    for superclass in cls.is_a:
        print(superclass)

# Print the superclasses and their types
def print_and_process_superclasses(ontology):
    for cls in ontology.classes():
        print(f"\nClass: {cls.name}")
        print("Superclasses and Restrictions:")
        complex_descriptions = []
        for sup in cls.is_a:
            if isinstance(sup, ThingClass):
                print(f"  - {sup.name}")
            else:
                complex_descriptions.append(sup)
                print("  - Some restriction or anonymous superclass")
        if complex_descriptions:
            process_complex_descriptions(cls, complex_descriptions)

# Define a function to process complex descriptions
def process_complex_descriptions(cls, descriptions):
    print(f"Processing complex descriptions for {cls.name}:")
    for desc in descriptions:
        print(f"  - {desc}")

# Use a reasoner to check consistency
with onto:
    inconsistent_classes = list(default_world.inconsistent_classes())
    print("Inconsistent classes:", inconsistent_classes)

    # Normalize subclass and equivalent axioms  
    def is_complex(description):
        return isinstance(description, (And, Or))
    
    # Define a function to normalize subclass axioms
    def normalize_subclass_axioms(ontology):
        # Create new classes for complex descriptions
        for cls in ontology.classes():
            # Check if the superclass is complex
            new_axioms = []
            for superclass in cls.is_a:
                if is_complex(superclass):
                    # Create a new class with a normalized name
                    new_class_name = f"Normalized_{cls.name}"
                    new_cls = types.new_class(new_class_name, (Thing,))
                    new_axioms.append((cls, superclass, new_cls))

            # Replace complex superclasses with new classes
            for cls, old_superclass, new_cls in new_axioms:
                cls.is_a.remove(old_superclass)
                cls.is_a.append(new_cls)
                new_cls.is_a.append(old_superclass)

    # Define a function to normalize equivalent axioms
    def normalize_equivalent_axioms(ontology):
        # Create new classes for complex descriptions
        for cls in ontology.classes():
            new_axioms = []
            # Check if the equivalent class is complex
            for equiv_cls in cls.equivalent_to:
                if is_complex(equiv_cls):
                    new_class_name = f"Equivalent_{cls.name}"
                    new_cls = types.new_class(new_class_name, (Thing,))
                    new_axioms.append((cls, equiv_cls, new_cls))
            # Replace complex equivalent classes with new classes
            for cls, old_equiv_cls, new_cls in new_axioms:
                cls.equivalent_to.remove(old_equiv_cls)
                cls.equivalent_to.append(new_cls)
                new_cls.is_a.append(old_equiv_cls)

    # Normalize subclass and equivalent axioms by generating unique names for new classes
    def generate_unique_name(cls):
        base_name = cls.name
        counter = 1
        while True:
            new_name = f"{base_name}_{counter}"
            if new_name not in onto.individuals():
                return new_name
            counter += 1

    # Define a function to replace anonymous individuals with named individuals
    def replace_anonymous_individuals(ontology):
        for cls in ontology.classes():
            for ind in cls.instances():
                if is_anonymous(ind):
                    print(f"Replacing anonymous individual: {ind}")
                    new_name = generate_unique_name(cls)
                    new_ind = cls(new_name)
                    for prop in ind.get_properties():
                        for value in prop[ind]:
                            prop[new_ind].append(value)
                    destroy_entity(ind)

    # Define a function to check if an individual is anonymous
    def is_anonymous(individual):
        return not individual.iri

    replace_anonymous_individuals(onto)

# Define functions to calculate various structural metrics

# Define a function to count classes
def count_classes(ontology):
    return len(list(ontology.classes()))

# Dfine a function to count object properties
def count_object_properties(ontology):
    return len(list(ontology.object_properties()))

# Define a function to count data properties
def count_data_properties(ontology):
    return len(list(ontology.data_properties()))

# Define a function to count individuals
def count_individuals(ontology):
    return len(list(ontology.individuals()))

# Define a function to count axioms
def count_axioms(ontology):
    class_axioms = sum(1 for _ in ontology.classes())
    object_property_axioms = sum(1 for _ in ontology.object_properties())
    data_property_axioms = sum(1 for _ in ontology.data_properties())
    individual_axioms = sum(1 for _ in ontology.individuals())
    return class_axioms + object_property_axioms + data_property_axioms + individual_axioms

# Define functions to calculate hierarchical metrics
def calculate_max_depth(ontology):
    def depth_of_inheritance_tree(cls, depth=0):
        subclasses = list(cls.subclasses())
        if not subclasses:
            return depth
        return max(depth_of_inheritance_tree(sub, depth + 1) for sub in subclasses)
    max_depths = (depth_of_inheritance_tree(cls) for cls in ontology.classes() if {Thing} == set(cls.ancestors()) - {cls})
    return max(max_depths, default=0)

# Define functions to calculate complexity metrics

# Define a function to count leaf classes
def count_leaf_classes(ontology):
    return sum(1 for cls in ontology.classes() if not list(cls.subclasses()))

# Define a function to calculate the average depth of the inheritance tree
def depth_of_inheritance_tree(cls, depth=0):
    subclasses = list(cls.subclasses())
    if not subclasses:
        return depth
    return max(depth_of_inheritance_tree(sub, depth + 1) for sub in subclasses)

# Define a function to calculate the average depth of the inheritance tree
def calculate_average_depth(ontology):
    total_depth = sum(depth_of_inheritance_tree(cls) for cls in ontology.classes())
    return total_depth / count_classes(ontology)

# Define a function to calculate the maximum width of the inheritance tree
def calculate_maximum_width(ontology):
    level_width = {}
    def traverse(cls, level):
        level_width[level] = level_width.get(level, 0) + 1
        for subclass in cls.subclasses():
            traverse(subclass, level + 1)

    top_level_classes = []
    for cls in ontology.classes():
        # Use .is_a to access direct superclasses, and filter Thing
        if all(isinstance(sup, ThingClass) or sup == Thing for sup in cls.is_a):
            top_level_classes.append(cls)

    for cls in top_level_classes:
        traverse(cls, 0)
    return max(level_width.values(), default=0)


# Define a function to calculate tangledness
def calculate_tangledness(ontology):
    # Count classes that have more than one direct superclass
    tangled_count = sum(1 for cls in ontology.classes()
                        if len([sup for sup in cls.is_a if isinstance(sup, owlready2.ThingClass)]) > 1)
    return tangled_count

# Define a function to calculate the ratio of leaf classes
def calculate_ratio_of_leaf_classes(ontology):
    return count_leaf_classes(ontology) / count_classes(ontology)

# Define a function to calculate the instance to class ratio
def calculate_instance_to_class_ratio(ontology):
    return count_individuals(ontology) / count_classes(ontology)

# Define a function to calculate annotation richness
def calculate_annotation_richness(ontology):
    annotations = 0
    for cls in ontology.classes():
        for prop in ontology.annotation_properties():
            values = cls.get_properties(prop)
            annotations += len(list(values)) if values else 0
    total_classes = count_classes(ontology)
    return annotations / total_classes if total_classes else 0

# Define a function to calculate subclass distribution
def calculate_subclass_distribution(ontology):
    distribution = {}
    for cls in ontology.classes():
        distribution[cls] = len(list(cls.subclasses()))
    return distribution

# Define a function to calculate property specificity
def property_specificity(ontology):
    specificity_scores = {}
    for prop in ontology.properties():
        # Check if domain and range are None before converting to list
        domain_specificity = len(list(prop.domain)) if prop.domain else 0
        range_specificity = len(list(prop.range)) if prop.range else 0
        specificity_scores[prop] = (domain_specificity, range_specificity)
    return specificity_scores


# Print the calculated metrics
print("Class Count:", count_classes(onto))
print("Object Property Count:", count_object_properties(onto))
print("Data Property Count:", count_data_properties(onto))
print("Individual Count:", count_individuals(onto))
print("Axiom Count:", count_axioms(onto))
print("Max Depth of Inheritance Tree:", calculate_max_depth(onto))
print("Leaf Class Count:", count_leaf_classes(onto))
print("Average Depth:", calculate_average_depth(onto))
print("Maximum Width:", calculate_maximum_width(onto))
print("Tangledness:", calculate_tangledness(onto))
print("Leaf Class Ratio:", calculate_ratio_of_leaf_classes(onto))
print("Instance to Class Ratio:", calculate_instance_to_class_ratio(onto))
print("Annotation Richness:", calculate_annotation_richness(onto))
print("Subclass Distribution:", calculate_subclass_distribution(onto))
print("Property Specificity:", property_specificity(onto))

# Calculated metrics (replace with your actual calculated values)
class_count = count_classes(onto)
object_property_count = count_object_properties(onto)
data_property_count = count_data_properties(onto)
individual_count = count_individuals(onto)
axiom_count = count_axioms(onto)
max_depth = calculate_max_depth(onto)
leaf_class_count = count_leaf_classes(onto)
average_depth = calculate_average_depth(onto)
maximum_width = calculate_maximum_width(onto)
tangledness = calculate_tangledness(onto)
leaf_class_ratio = calculate_ratio_of_leaf_classes(onto)
instance_to_class_ratio = calculate_instance_to_class_ratio(onto)
annotation_richness = calculate_annotation_richness(onto)
subclass_distribution = calculate_subclass_distribution(onto)
property_specificity = property_specificity(onto)

# Define the metrics to visualize
counts = [count_classes(onto), count_object_properties(onto), count_data_properties(onto), individual_count]
hierarchical_metrics = [max_depth, average_depth, maximum_width]
complexity_metrics = [tangledness, leaf_class_ratio]
richness_metrics = [instance_to_class_ratio, annotation_richness]

# Titles and labels for the charts
titles = ['Ontology Element Counts', 'Hierarchical Metrics', 'Class Complexity Metrics', 'Ontology Richness Metrics']
x_labels = [
    ['Classes', 'Object Properties', 'Data Properties', 'Individuals'],
    ['Max Depth', 'Average Depth', 'Max Width'],
    ['Tangledness', 'Leaf Class Ratio'],
    ['Instance to Class Ratio', 'Annotation Richness']
]
y_label = 'Value'

# Choose a nice color palette
colors = sns.color_palette('pastel')

# First figure with top plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(x_labels[0], counts, color=colors)
plt.title(titles[0])
plt.ylabel(y_label)

plt.subplot(1, 2, 2)
plt.bar(x_labels[1], hierarchical_metrics, color=colors)
plt.title(titles[1])
plt.ylabel(y_label)

# Despine the top and right borders
sns.despine()

plt.tight_layout()
plt.show()

# Separate figures for the rest of the plots
for i, metrics in enumerate([complexity_metrics, richness_metrics], start=2):
    plt.figure(figsize=(6, 6))
    plt.bar(x_labels[i], metrics, color=colors[:len(metrics)])
    plt.title(titles[i])
    plt.ylabel(y_label)
    sns.despine()
    plt.tight_layout()
    plt.show()

# Filter out zero values from subclass distribution and extract class names as strings
filtered_subclass_distribution = {cls.name: count for cls, count in calculate_subclass_distribution(onto).items() if count > 0}

# Prepare data for the Property Specificity scatter plot
# Filter out entries where both domain and range specificities are zero and convert property objects to their names
filtered_property_specificity = {prop.name: (domain, range_) for prop, (domain, range_) in property_specificity.items() if domain > 0 or range_ > 0}

# Plotting Subclass Distribution
plt.figure(figsize=(10, 8))
plt.barh(list(filtered_subclass_distribution.keys()), list(filtered_subclass_distribution.values()), color=sns.color_palette("husl", len(filtered_subclass_distribution)))
plt.xlabel('Number of Subclasses')
plt.title('Subclass Distribution')
sns.despine(left=True, bottom=True)  # Remove the top and right spines from plot
plt.tight_layout()
plt.show()

# Convert property objects to string names and pair with their specificity values
properties_with_specificity = [(prop.name, *spec) for prop, spec in property_specificity.items()]
# Sort properties by domain then range specificity in descending order
sorted_properties = sorted(properties_with_specificity, key=lambda x: (x[1], x[2]), reverse=True)
# Print out properties with the highest domain and range specificity
highest_domain_specificity = sorted_properties[0][1]
highest_range_specificity = sorted_properties[0][2]

# Print the properties with the highest domain and range specificity
print("Properties with the highest domain specificity:")
for name, dom_spec, _ in sorted_properties:
    if dom_spec == highest_domain_specificity:
        print(f"{name} (Domain Specificity: {dom_spec})")

print("\nProperties with the highest range specificity:")
for name, _, ran_spec in sorted_properties:
    if ran_spec == highest_range_specificity:
        print(f"{name} (Range Specificity: {ran_spec})")
