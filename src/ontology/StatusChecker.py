# Import the necessary libarary that allow us to work in the ontolgoy
from owlready2 import *

# Loading the predefined ontology
onto = get_ontology('csonto/target/csonto/src/ontology/csonto-edit.rdf').load()

# Function to update the weights based on the status of the policies
def update_has_weight_for_policies():
    # Filter classes that contain '_Policies' in their name
    classes_containing_policies = [cls for cls in onto.classes() if '_Policies' in cls.name]
    
    # Iterate through the filtered classes and their individuals
    for cls in classes_containing_policies:
        for individual in cls.instances():
            
            status = individual.status # Get the 'status' of the individual
            
            # Check the 'status' and update 'HasWeight' accordingly
            if status in ["Bypassed", "Ignored", "Violated"]:
                individual.HasWeight = [0]  # Set 'HasWeight' to 0 for these statuses
            elif status == "Implemented":
                # If 'status' is "Implemented", we assume 'HasWeight' remains unchanged
                # and do nothing, or you can reinforce the current value or logic here
                pass
            else:
                # Setting the deafult to be implemented
                individual.status = ["Implemented"]  

# Execute the function to update 'HasWeight' values for policy-related classes
update_has_weight_for_policies()

# Save the updated ontology
onto.save(file="csonto/target/csonto/src/ontology/csonto-edit.rdf")

print("Status is checked and updated successfully!")