# Import the necessary libarary that allow us to work in the ontolgoy
from owlready2 import *

# Loading the predefined ontology
onto = get_ontology('/Users/alisami/Desktop/csonto-edit.rdf').load()

# Define the RiskManagement class with appropriate data properties
with onto:
    # Definning new class for managing new cases for the risk management
    class RiskManagementCases(onto.Risk_Management):
        pass

    class CaseID(DataProperty):
        domain = [RiskManagementCases]
        range = [str]

    class CaseName(DataProperty):
        domain = [RiskManagementCases]
        range = [str]

    class RiskList(DataProperty):
        domain = [RiskManagementCases]
        range = [str]  

    class Likelihood(DataProperty):
        domain = [RiskManagementCases]
        range = [str]  

    class Consequences(DataProperty):
        domain = [RiskManagementCases]
        range = [str]  

    class RiskEffects(DataProperty):
        domain = [RiskManagementCases]
        range = [str]

    class RiskSolutions(DataProperty):
        domain = [RiskManagementCases]
        range = [str]

    class RiskDecisions(DataProperty):
        domain = [RiskManagementCases]
        range = [str]

    class PolicyAvailability(DataProperty):
        domain = [RiskManagementCases]
        range = [str]  
    # Define the vulnerability class lits for each vulnerability
    class VulnerabilityList(onto.Vulnerability_And_Web_Management):
        pass
   # Define the patch class lits for each vulnerability patch
    class PatchList(onto.Vulnerability_And_Web_Management):
        pass

    # Define data properties for the Vulnerability class
    class vulnerabilityID(DataProperty):
        domain = [VulnerabilityList]
        range = [str]
    
    class vulnerabilityName(DataProperty):
        domain = [VulnerabilityList]
        range = [str]
    
    class vulnerabilityDescription(DataProperty):
        domain = [VulnerabilityList]
        range = [str]
    
    class vulnerabilityEffects(DataProperty):
        domain = [VulnerabilityList]
        range = [str]

    # Define data properties for the Patch class
    class patchID(DataProperty):
        domain = [PatchList]
        range = [str]
    
    class patchName(DataProperty):
        domain = [PatchList]
        range = [str]
    
    class patchReleaseDate(DataProperty):
        domain = [PatchList]
        range = [str]  
    
    # Define the object property to link Vulnerability to Patch
    class hasPatch(ObjectProperty):
        domain = [VulnerabilityList]
        range = [PatchList]

#with onto:
#    class Status(DataProperty):
#       range = [str]
# Define or retrieve the 'status' data property with a range of xsd:string
#status_property = onto.search_one(iri="*status")

# Filter classes that contain '_Policies' in their name
#classes_containing_policies = [cls for cls in onto.classes() if '_Policies' in cls.name]

# Iterate through the filtered classes and their individuals
#for cls in classes_containing_policies:
    #for individual in cls.instances():
        # Use .append() if you want to add multiple statuses or ensure the property can hold multiple values
        # If each individual should only have a single status value, use direct assignment:
        # individual.status = ["Active"]  # This sets 'Active' as the status, replacing any existing status
        #if not individual.Status:  # Check if the individual doesn't already have a status
            #individual.Status.append("Active")  # Set a default status value

# Save the modified ontology to a file
onto.save(file="/Users/alisami/Desktop/csonto-edit.rdf")

# Print a success message
#print("Successfully added/updated the 'status' property for all individuals in classes containing '_Policies'.")