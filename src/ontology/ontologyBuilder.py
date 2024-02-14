# This scripts read from the Secure Controls Framework (SCF) Excel file and
# generates instances of the different domain classes in the ontology.
# These controls defined by the NIST and we made it as a part of our ontology
# It will helpe to collect the data and the overall score of the organization.


# Import the necessary libraries
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD, RDFS
import pandas as pd

# Load the Excel file
excel_file_path = '/Users/alisami/Desktop/FYP/final-year-project-ASHS21/SCF-OneSheet.xlsx'
df = pd.read_excel(excel_file_path)

# Initialize the RDF graph
g = Graph()
ontology_path = '/Users/alisami/Desktop/FYP/final-year-project-ASHS21/Test.rdf'  # Update this to the path of your ontology
 

# Define your ontology's namespace
my_ns = Namespace("http://FYP-ASHS21/csonto/")


# Define the memeber of the classes that related to the parent class

#PartentDomainClasslist = ["Change Management","Risk Management", "Third-Party Management", "Technology Development & Acquisition"
#               , "Secure Engineering & Architecture"]

# Iterating through the DataFrame and adding the data to the ontology as instances of the different domain classes

#Controls_df = df[df['SCF Domain'].str.contains('|'.join(PartentDomainClasslist), na=False)]

# Define the Control class
#ParentDomainClass_Policies = my_ns.ParentDomainClass_Policies

# Iterate through the DataFrame and add the data to the ontology as instances of ParentClass_Policies
for index, row in Controls_df.iterrows():
    try:
        prefix = "Prefix-" if row['SCF Control'] != "Parent Class Name" else ""
        
        # Generate instance ID with or without prefix based on the control name
        instance_id = "{}-{}".format(prefix + row['SCF #'], row['SCF Control']).replace(' ', '').replace('/', '-').replace('(', '-').replace(')', '-').replace(',','-')
        instance_id = instance_id.replace('\n', '').replace(' ', '')  # Replace newline characters and spaces
        
        instance_uri = URIRef(my_ns[instance_id])

    # Add the individual to the graph as a member of the ParentClass_Policies class
        g.add((instance_uri, RDF.type, ParentDomainClass_Policies))

    
    
    # Add description as rdf:comment
        if pd.notnull(row['Secure Controls Framework (SCF)\nControl Description']):
            g.add((instance_uri, RDFS.comment, Literal(row['Secure Controls Framework (SCF)\nControl Description'], datatype=XSD.string)))
    
    # Add the weight using the 'HasWeight' property
        if pd.notnull(row['Relative Control Weighting']):
            g.add((instance_uri, my_ns.HasWeight, Literal(row['Relative Control Weighting'], datatype=XSD.integer)))
    except Exception as e:
        print(f"Error processing row {index}: {e}")
    

# Save the updated graph back to the ontology file
updated_ontology_path = 'Path-To-File'  
g.serialize(destination=updated_ontology_path, format='xml')
