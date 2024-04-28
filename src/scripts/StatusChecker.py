# Import the necessary libarary that allow us to work in the ontolgoy
import owlready2 as owl
import random

# Loading the predefined ontology
onto = owl.get_ontology('/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/csonto-edit.rdf').load()

# Function to update the weights based on the status of the policies
def update_has_weight_for_policies():
    status_options = ["Bypassed", "Ignored", "Violated", "Implemented"]
    
    # Filter classes that contain '_Policies' in their name
    classes_containing_policies = [cls for cls in onto.classes() if '_Policies' in cls.name]
    
    # Iterate through the filtered classes and their individuals
    for cls in classes_containing_policies:
        for individual in cls.instances():
            # Randomly assign a status from the predefined list
            status = random.choice(status_options)
            individual.Status = [status]  # Assuming 'Status' can be directly assigned like this
            
            # Update 'HasWeight' based on the randomly assigned 'Status'
            if status in ["Bypassed", "Ignored", "Violated"]:
                individual.HasWeight = [0]  # Set 'HasWeight' to 0 for these statuses
            elif status == "Implemented":
                # If 'status' is "Implemented", the 'HasWeight' might need specific handling
                # Here we leave 'HasWeight' unchanged assuming it's set correctly elsewhere or previously
                pass
            # No need for an else clause if we're sure statuses only come from 'status_options'

# Execute the function to update 'HasWeight' values for policy-related classes
update_has_weight_for_policies()

# Save the updated ontology
onto.save(file="/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/csonto-edit.rdf")

print("Status is checked and updated successfully!")