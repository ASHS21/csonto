from owlready2 import *

# Load your ontology
onto = get_ontology("/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/csonto-edit.rdf").load()
sync_reasoner_pellet()
for cls in onto.classes():
    for superclass in cls.is_a:
        print(superclass)

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

def process_complex_descriptions(cls, descriptions):
    print(f"Processing complex descriptions for {cls.name}:")
    for desc in descriptions:
        print(f"  - {desc}")

# Use a reasoner to check consistency
with onto:
    inconsistent_classes = list(default_world.inconsistent_classes())
    print("Inconsistent classes:", inconsistent_classes)

    def is_complex(description):
        return isinstance(description, (And, Or))

    def normalize_subclass_axioms(ontology):
        for cls in ontology.classes():
            new_axioms = []
            for superclass in cls.is_a:
                if is_complex(superclass):
                    new_class_name = f"Normalized_{cls.name}"
                    new_cls = types.new_class(new_class_name, (Thing,))
                    new_axioms.append((cls, superclass, new_cls))
            for cls, old_superclass, new_cls in new_axioms:
                cls.is_a.remove(old_superclass)
                cls.is_a.append(new_cls)
                new_cls.is_a.append(old_superclass)

    def normalize_equivalent_axioms(ontology):
        for cls in ontology.classes():
            new_axioms = []
            for equiv_cls in cls.equivalent_to:
                if is_complex(equiv_cls):
                    new_class_name = f"Equivalent_{cls.name}"
                    new_cls = types.new_class(new_class_name, (Thing,))
                    new_axioms.append((cls, equiv_cls, new_cls))
            for cls, old_equiv_cls, new_cls in new_axioms:
                cls.equivalent_to.remove(old_equiv_cls)
                cls.equivalent_to.append(new_cls)
                new_cls.is_a.append(old_equiv_cls)

    def generate_unique_name(cls):
        base_name = cls.name
        counter = 1
        while True:
            new_name = f"{base_name}_{counter}"
            if new_name not in onto.individuals():
                return new_name
            counter += 1

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

    def is_anonymous(individual):
        return not individual.iri

    replace_anonymous_individuals(onto)

# Define functions to calculate various structural metrics
def count_classes(ontology):
    return len(list(ontology.classes()))

def count_object_properties(ontology):
    return len(list(ontology.object_properties()))

def count_data_properties(ontology):
    return len(list(ontology.data_properties()))

def count_individuals(ontology):
    return len(list(ontology.individuals()))

def count_axioms(ontology):
    class_axioms = sum(1 for _ in ontology.classes())
    object_property_axioms = sum(1 for _ in ontology.object_properties())
    data_property_axioms = sum(1 for _ in ontology.data_properties())
    individual_axioms = sum(1 for _ in ontology.individuals())
    return class_axioms + object_property_axioms + data_property_axioms + individual_axioms

def calculate_max_depth(ontology):
    def depth_of_inheritance_tree(cls, depth=0):
        subclasses = list(cls.subclasses())
        if not subclasses:
            return depth
        return max(depth_of_inheritance_tree(sub, depth + 1) for sub in subclasses)
    max_depths = (depth_of_inheritance_tree(cls) for cls in ontology.classes() if {Thing} == set(cls.ancestors()) - {cls})
    return max(max_depths, default=0)

def count_leaf_classes(ontology):
    return sum(1 for cls in ontology.classes() if not list(cls.subclasses()))

def depth_of_inheritance_tree(cls, depth=0):
    subclasses = list(cls.subclasses())
    if not subclasses:
        return depth
    return max(depth_of_inheritance_tree(sub, depth + 1) for sub in subclasses)

def calculate_average_depth(ontology):
    total_depth = sum(depth_of_inheritance_tree(cls) for cls in ontology.classes())
    return total_depth / count_classes(ontology)

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


def calculate_tangledness(ontology):
    # Count classes that have more than one direct superclass
    tangled_count = sum(1 for cls in ontology.classes()
                        if len([sup for sup in cls.is_a if isinstance(sup, owlready2.ThingClass)]) > 1)
    return tangled_count

def calculate_ratio_of_leaf_classes(ontology):
    return count_leaf_classes(ontology) / count_classes(ontology)

def calculate_instance_to_class_ratio(ontology):
    return count_individuals(ontology) / count_classes(ontology)

def calculate_annotation_richness(ontology):
    annotations = 0
    for cls in ontology.classes():
        for prop in ontology.annotation_properties():
            values = cls.get_properties(prop)
            annotations += len(list(values)) if values else 0
    total_classes = count_classes(ontology)
    return annotations / total_classes if total_classes else 0


def calculate_subclass_distribution(ontology):
    distribution = {}
    for cls in ontology.classes():
        distribution[cls] = len(list(cls.subclasses()))
    return distribution

def property_specificity(ontology):
    specificity_scores = {}
    for prop in ontology.properties():
        # Check if domain and range are None before converting to list
        domain_specificity = len(list(prop.domain)) if prop.domain else 0
        range_specificity = len(list(prop.range)) if prop.range else 0
        specificity_scores[prop] = (domain_specificity, range_specificity)
    return specificity_scores



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









