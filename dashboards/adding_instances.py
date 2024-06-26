# The scrpit below is used to add instances to the ontology 
# It generates mock data for the asset management and risk management classes
# The mock data is then added to the ontology

# Import required libraries
import pandas as pd
import numpy as np
import random
import string
from owlready2 import get_ontology
import uuid

# Helper functions for generating mock data
def random_string(length=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(length))

# Function to randomly select a city from the list
def random_city(cities):
    return random.choice(cities)

# List of cities for 'LastKnownLocation' and 'OriginalPlace'
cities = ['New York', 'London', 'Tokyo', 'Sydney', 'Paris', 'Berlin', 'Singapore', 'Dubai', 'San Francisco', 'Sao Paulo']

# Generate mock data
num_assets = 100  # Number of assets to generate
data = {
    'AssetID': [random_string(8) for _ in range(num_assets)],
    'Name': [f"{asset_type}-{random_string(5)}" for asset_type in np.random.choice(['Server', 'Workstation', 'Router', 'Switch', 'Firewall'], size=num_assets)],
    'TypeOfAsset': np.random.choice(['Hardware', 'Software', 'Network Device'], size=num_assets),
    'LastKnownLocation': [random_city(cities) for _ in range(num_assets)],
    'OriginalPlace': [random_city(cities) for _ in range(num_assets)]
}

# Create DataFrame from mock data
asset_management_df = pd.DataFrame(data)

# Load the ontology
onto = get_ontology("/workspaces/csonto/dashboards/csonto-edit.rdf").load()

# Access the AssetsList class using the provided IRI
AssetsList = onto.search_one(iri="http://FYP-ASHS21/csonto#AssetsList")

# Verify the AssetsList class was found
if not AssetsList:
    raise ValueError("AssetsList class not found in the ontology")

# Iterate over the DataFrame rows and create instances of the AssetsList class
for index, row in asset_management_df.iterrows():
    asset_instance = AssetsList()
    
    # Assign properties to the instance, ensuring they are treated as data properties
    # Since the properties are not functional, you cannot assign directly.
    # Instead, you need to use append or set method.
    getattr(asset_instance, "originalPlace").append(str(row['OriginalPlace']))
    getattr(asset_instance, "typeOfAsset").append(str(row['TypeOfAsset']))
    getattr(asset_instance, "lastKnownLocation").append(str(row['LastKnownLocation']))

    # For functional properties, you can assign directly
    asset_instance.name = str(row['Name'])  # Assuming 'name' is a functional property
    asset_instance.AssetID = str(row['AssetID'])  # Assuming 'AssetID' is a functional property

# Function to create instances of RiskManagementCases
def create_risk_management_cases(n_instances):
    risk_management_cases = onto.search_one(iri="http://FYP-ASHS21/csonto#RiskManagementCases")
    
    for i in range(n_instances):
        new_case = risk_management_cases(f"RiskManagementCase_{uuid.uuid4()}")

        # Assuming CaseID and other properties can have only a single value per individual
        # and are correctly modeled as functional properties in your ontology
        new_case.CaseID.append(str(uuid.uuid4()))

         # Single value for functional property
        new_case.CaseName.append(f"CaseName_{i}")
        new_case.Consequences.append(random.choice(["Low", "Medium", "High"]))
        new_case.Likelihood.append(random.choice(["Unlikely", "Possible", "Likely", "Certain"]))
        new_case.PolicyAvailability.append(random.choice([True, False]))
        new_case.RiskDecisions.append(random.choice(["Accept", "Transfer", "Reject", "Ignore"]))
        
        # For non-functional properties expecting multiple values, use .append() for each value
        # or assign a flat list of values directly
        new_case.RiskEffects.append(f"Effect_{random.randint(1, 5)}")
        new_case.RiskList.append(f"Risk_{random.randint(1, 10)}")
        new_case.RiskSolutions.append(f"Solution_{random.randint(1, 3)}")


# Function to create instances of VulnerabilityList and PatchList
def create_mock_data(num_vulnerabilities=100, num_patches=35):
    # Create Vulnerability instances
    for i in range(1, num_vulnerabilities + 1):
        vulnerability = onto.VulnerabilityList()  # Instantiate a new VulnerabilityList object
        
        # Since properties are not functional, use list assignment or append
        vulnerability.vulnerabilityDescription = [f"Description of vulnerability {i}"]
        vulnerability.vulnerabilityEffects = [f"Effects of vulnerability {i}"]
        vulnerability.vulnerabilityID = [f"VULN{i:03d}"]
        vulnerability.vulnerabilityName = [f"Vulnerability_{i}"]

    # Create Patch instances
    for i in range(1, num_patches + 1):
        patch = onto.PatchList()  # Instantiate a new PatchList object
        
        # Since properties are not functional, use list assignment or append
        patch.patchID = [f"PATCH{i:03d}"]
        patch.patchName = [f"Patch_{i}"]
        patch.patchReleaseDate = [f"2024-03-{i:02d}"]

    # Randomly link patches to vulnerabilities
    all_vulnerabilities = list(onto.VulnerabilityList.instances())
    all_patches = list(onto.PatchList.instances())

    for patch in all_patches:
        linked_vulnerabilities = random.sample(all_vulnerabilities, k=random.randint(1, 5))
        for vulnerability in linked_vulnerabilities:
            # Use append() for non-functional object properties
            if not hasattr(patch, 'ProvidePatch'):
                patch.ProvidePatch = [vulnerability]
            else:
                patch.ProvidePatch.append(vulnerability)
            
            if not hasattr(vulnerability, 'hasPatch'):
                vulnerability.hasPatch = [patch]
            else:
                vulnerability.hasPatch.append(patch)

# Call the function to create mock
create_mock_data()

# Save the updated ontology
onto.save(file="/workspaces/csonto/dashboards/csonto-edit.rdf")
