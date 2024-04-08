# This script is showing the score dashboard, which is used to display the knowledge graph visualization and other related information.
# The script uses the ontology to extract vulnerabilities, risks, and policies data and insert it into Neo4j.
# It also calculates and updates the weights of the policies based on the risk adjustments.
# The script shows the different classes of cybersecurity organizational policies and their status.

# Importing required libraries
from owlready2 import get_ontology
import pandas as pd
import streamlit as st
import altair as alt
import random
from neo4j import GraphDatabase
import json
from pyvis.network import Network
import streamlit.components.v1 as components


# Neo4j connection class
class Neo4jConnection:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.driver = GraphDatabase.driver(self.config['neo4j']['uri'], 
                                           auth=(self.config['neo4j']['user'], 
                                                 self.config['neo4j']['password']))
    
    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as file:
            return json.load(file)
    
    def close(self):
        if self.driver:
            self.driver.close()

# Function to calculate and update the weights of the policies
def calculate_and_update_weights(onto_path, risk_adjustments=None):
    onto = get_ontology(onto_path).load()

    classes_iris = [
        "http://FYP-ASHS21/csonto/Asset_Management_Policies",
        "http://FYP-ASHS21/csonto/Assurance_Policies",
        "http://FYP-ASHS21/csonto/Change_Management_Policies",
        "http://FYP-ASHS21/csonto/Emerging_Technologies_Policies",
        "http://FYP-ASHS21/csonto/Governance_Policies",
        "http://FYP-ASHS21/csonto/Security_Operations_Policies"
    ]

    total_weight_before, total_weight_after = 0, 0

    for class_iri in classes_iris:
        policy_class = onto.search_one(iri=class_iri)
        if policy_class:
            for instance in policy_class.instances():
                weight = getattr(instance, "HasWeight", [0])[0]
                total_weight_before += weight

                if risk_adjustments and instance.name in risk_adjustments:
                    adjusted_weight = adjust_weight_based_on_risk(weight, risk_adjustments[instance.name])
                else:
                    adjusted_weight = assign_new_status_and_adjust_weight(instance, weight)

                instance.HasWeight = [adjusted_weight]
                total_weight_after += adjusted_weight

    return total_weight_before, total_weight_after

# Function to get the unpatched vulnerabilities from the ontology
def extract_unpatched_vulnerabilities(onto_path):
    # Get all vulnerabilities
    onto = get_ontology(onto_path).load()
    # Search for the VulnerabilityList class using the provided IRI
    vulnerability_class = onto.search_one(iri="http://FYP-ASHS21/csonto#VulnerabilityList")
    vulnerabilities = vulnerability_class.instances()
    
    # Get all instances of the PatchList class using the provided IRI
    patch_class = onto.search_one(iri="http://FYP-ASHS21/csonto#PatchList")
    patches = patch_class.instances()

    # Extract all the vulnerability IDs that have been patched
    patched_vuln_ids = set()
    for patch in patches:
        for vuln in patch.ProvidePatch:
            # Extract the first item from the list of IDs
            vuln_id = next(iter(vuln.vulnerabilityID), None)
            if vuln_id is not None:
                patched_vuln_ids.add(vuln_id)

    # Filter out vulnerabilities that have not been patched
    unpatched_vulnerabilities = []
    for vuln in vulnerabilities:
        # Extract the first item from the list of IDs
        vuln_id = next(iter(vuln.vulnerabilityID), None)
        if vuln_id is not None and vuln_id not in patched_vuln_ids:
            unpatched_vulnerabilities.append({
                'vulnerabilityID': vuln_id,
                'vulnerabilityName': next(iter(vuln.vulnerabilityName), "N/A"),
                'vulnerabilityDescription': next(iter(vuln.vulnerabilityDescription), "N/A"),
                'vulnerabilityEffects': next(iter(vuln.vulnerabilityEffects), "N/A")
            })

    return unpatched_vulnerabilities

# Function to extract critical accepted risks from the ontology
def extract_critical_accepted_risks(onto_path):
    # Get all risks
    onto = get_ontology(onto_path).load()
    risk_cases_class = onto.search_one(iri="http://FYP-ASHS21/csonto#RiskManagementCases")
    risks = risk_cases_class.instances()

    # Filter risks based on the criteria
    critical_risks = [{
        'CaseID': risk.CaseID[0] if risk.CaseID else "N/A",
        'CaseName': risk.CaseName[0] if risk.CaseName else "N/A",
        'Consequences': risk.Consequences[0] if risk.Consequences else "N/A",
        'Likelihood': risk.Likelihood[0] if risk.Likelihood else "N/A",
        'RiskDecisions': risk.RiskDecisions[0] if risk.RiskDecisions else "N/A",
        'RiskEffects': risk.RiskEffects[0] if risk.RiskEffects else "N/A",
        # Ensure this checks if a policy is available for the risk
        'PolicyAvailable': getattr(risk, 'PolicyAvailable', [False])[0]
    } for risk in risks if risk.RiskDecisions and risk.RiskDecisions[0] == "Accept" and
       risk.Likelihood and risk.Likelihood[0] == "Certain" and 
       risk.Consequences and risk.Consequences[0] == "High"]

    return critical_risks

# Function to extract non-compliant policies from the ontology
def extract_non_compliant_policies(onto_path):
    onto = get_ontology(onto_path).load()
    
    non_compliant_policies = []
    
    classes_containing_policies = [cls for cls in onto.classes() if '_Policies' in cls.name]

    
    for cls in classes_containing_policies:
        for individual in cls.instances():
            if individual.Status and individual.Status[0] != "Implemented":
                non_compliant_policies.append({
                    'PolicyID': individual.name,
                    'HasWeight': individual.HasWeight[0] if individual.HasWeight else 0,
                    'Status': individual.Status[0]
                })
    
    return non_compliant_policies

# Function to insert data into Neo4j
def insert_data_into_neo4j(neo4j_conn, risks, vulnerabilities, policies):
    with neo4j_conn.driver.session() as session:
        # Insert vulnerabilities into Neo4j
        for vuln in vulnerabilities[:5]:
            session.run("""
                MERGE (v:Vulnerability {id: $id})
                ON CREATE SET v.name = $name, v.description = $description, v.effects = $effects
                MERGE (s:Score {value: 'Low'})
                MERGE (s)-[:CAUSED_BY]->(v)
                """, 
                id=vuln['vulnerabilityID'], 
                name=vuln['vulnerabilityName'], 
                description=vuln['vulnerabilityDescription'], 
                effects=vuln['vulnerabilityEffects'])

        # Insert risks into Neo4j
        for risk in risks[:5]:
            session.run("""
                MERGE (r:Risk {id: $id})
                ON CREATE SET r.name = $name, r.effects = $effects, r.decisions = $decisions
                MERGE (s:Score {value: 'Low'})
                MERGE (s)-[:CAUSED_BY]->(r)
                """, 
                id=risk['CaseID'], 
                name=risk['CaseName'], 
                effects=risk['RiskEffects'], 
                decisions=risk['RiskDecisions'])

        # Insert policies into Neo4j
        for policy in policies[:5]:
        # Check if 'HasWeight' is 0 and set label to 'PolicyID' if true, otherwise, use the 'HasWeight'
           
            session.run("""
             MERGE (p:Policy {id: $id})
            ON CREATE SET p.id = $id, p.status = $status
            MERGE (s:Score {value: 'Low'})
            MERGE (s)-[:CAUSED_BY]->(p)
            """, 
            id=policy['PolicyID'], 
            status=policy['Status'])

# Function to adjust the weight based on the risk
def adjust_weight_based_on_risk(weight, adjustment_factor):
    return weight * adjustment_factor

# Function to assign a new status and adjust the weight
def assign_new_status_and_adjust_weight(instance, weight):
    new_status = random.choice(["Implemented", "Violated", "Bypassed", "Ignored"])
    instance.Status = [new_status]
    adjusted_weight = weight if new_status == "Implemented" else 0
    return adjusted_weight

# Function to extract risk information from the ontology
def extract_risk_information(onto_path):
    onto = get_ontology(onto_path).load()
    
    risk_cases_class = onto.search_one(iri="http://FYP-ASHS21/csonto#RiskManagementCases")
    risks_data = []

    if risk_cases_class:
        for risk_case in risk_cases_class.instances():
            risks_data.append({
                'CaseID': getattr(risk_case, "CaseID", ["Unknown"])[0],
                'CaseName': getattr(risk_case, "CaseName", ["Unknown"])[0],
                'Consequences': getattr(risk_case, "Consequences", ["Unknown"])[0],
                'Likelihood': getattr(risk_case, "Likelihood", ["Unknown"])[0],
                'RiskDecisions': getattr(risk_case, "RiskDecisions", ["Unknown"])[0],
                'RiskList': getattr(risk_case, "RiskList", ["Unknown"])[0],
                'RiskSolutions': getattr(risk_case, "RiskSolutions", ["Unknown"])[0],
                'RiskEffects': getattr(risk_case, "RiskEffects", ["Unknown"])[0],
                'PolicyAvailability': getattr(risk_case, "PolicyAvailability", [False])[0]
            })

    return pd.DataFrame(risks_data)

# Function to extract policy data from the ontology
def extract_policy_data(onto_path):
    onto = get_ontology(onto_path).load()

    classes_iris = [
        "http://FYP-ASHS21/csonto/Asset_Management_Policies",
        "http://FYP-ASHS21/csonto/Assurance_Policies",
        "http://FYP-ASHS21/csonto/Change_Management_Policies",
        "http://FYP-ASHS21/csonto/Emerging_Technologies_Policies",
        "http://FYP-ASHS21/csonto/Governance_Policies",
        "http://FYP-ASHS21/csonto/Security_Operations_Policies"
    ]

    policies_data = []

    for class_iri in classes_iris:
        policy_class = onto.search_one(iri=class_iri)
        if policy_class:
            for instance in policy_class.instances():
                policies_data.append({
                    'Class': policy_class.name,
                    'Status': instance.Status[0] if instance.Status else "Unknown",
                    'Weight': instance.HasWeight[0] if instance.HasWeight else 0
                })
                
    return pd.DataFrame(policies_data)

# Function to apply consequence adjustments to the risks
def apply_consequence_adjustments(df_risks):
    adjustments = {}
    for _, risk in df_risks.iterrows():
        if risk['Consequences'] == 'Medium':
            adjustments[risk['CaseName']] = 0.75  # Reduce by 25%
        elif risk['Consequences'] == 'High':
            adjustments[risk['CaseName']] = 0.65  # Reduce by 35%
        else:
            adjustments[risk['CaseName']] = 1  # No change
    return adjustments

# Function to visualize the graph
def visualize_graph(neo4j_conn):
    # Create a new network graph
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')

    # Fetch data from Neo4j
    with neo4j_conn.driver.session() as session:
        results = session.run("""
        MATCH (e)<-[r:CAUSED_BY]-(s:Score {value: 'Low'})
        WHERE e:Vulnerability OR e:Risk OR e:Policy
        RETURN e, r, s
        """)
        
        for record in results:
            e = record['e']
            s = record['s']
            # Use the _id attribute or call id(e) and id(s) in the RETURN clause of your Cypher query
            e_id = str(e._id)
            s_id = str(s._id)
            
            # Add entities to the graph, checking the labels to determine the color
            if 'Vulnerability' in e.labels:
                net.add_node(e_id, label=e['name'], color='red')
            elif 'Risk' in e.labels:
                net.add_node(e_id, label=e['name'], color='blue')
            elif 'Policy' in e.labels:
                net.add_node(e_id, label=e['name'], color='green')

            # Add the Score node to the graph if it doesn't already exist
            net.add_node(s_id, label=s['value'], color='orange', size=15)
            
            # Add a directed edge from entity to Score
            net.add_edge(e_id, s_id)

    # Generate the network in HTML format
    net.show('graph.html', notebook=False)
    return 'graph.html'

# Main function to render the Streamlit app
def app():

    config_path = '/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/pages/config.json'  
    
    # Initialize connection
    conn = Neo4jConnection(config_path)
    st.title('Policy and Risk Management Dashboard')
    onto_path = "/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/csonto-edit.rdf"
    html_file_path = visualize_graph(conn)

    st.title('Knowledge Graph Visualization')
    html_file = open(html_file_path, 'r', encoding='utf-8')
    source_code = html_file.read() 
    components.html(source_code, width=800, height=600)
    
    # Extract information
    vulnerabilities_without_patches = extract_unpatched_vulnerabilities(onto_path)
    critical_accepted_risks = extract_critical_accepted_risks(onto_path)
    extracted_policies = extract_non_compliant_policies(onto_path)
    df_policies = extract_policy_data(onto_path)
    df_risks = extract_risk_information(onto_path)
    
    # Display extracted information in the app for verification
    st.subheader("Unpatched Vulnerabilities")
    st.write(vulnerabilities_without_patches)

    st.subheader("Critical Accepted Risks")
    st.write(critical_accepted_risks)

    # If a button is clicked, insert the data into Neo4j
    if st.button('Update Neo4j Graph'):
        insert_data_into_neo4j(conn, critical_accepted_risks, vulnerabilities_without_patches, extracted_policies)
        st.success("Neo4j Graph updated with new vulnerabilities, risks, and policies!")

    # Calculate and update the weights of the policies
    if 'weight_before' not in st.session_state:
        st.session_state['weight_before'], _ = calculate_and_update_weights(onto_path)
        st.session_state['weight_after'] = st.session_state['weight_before']

    st.metric(label="Initial Total Weight Across All Classes", value=st.session_state['weight_before'])

    # Recalculate the weights if the button is clicked
    if st.button('Recalculate Weights'):
        _, st.session_state['weight_after'] = calculate_and_update_weights(onto_path)
        st.metric(label="Total Weight Across All Classes After Changes", value=st.session_state['weight_after'])
        if st.session_state['weight_before'] > 0:
            percentage_change = (st.session_state['weight_after'] / st.session_state['weight_before']) * 100
            st.metric(label="Percentage of Initial Total Weight", value=f"{percentage_change:.2f}%")
        else:
            st.error("Initial total weight is zero, cannot calculate percentage.")

    
    # Display the extracted policies and risks
    status_counts = df_policies['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Counts']
    chart1 = alt.Chart(status_counts).mark_bar().encode(
        x='Status',
        y='Counts',
        color='Status:N',
        tooltip=['Status', 'Counts']
    ).properties(title='Policy Counts by Status')
    st.altair_chart(chart1, use_container_width=True)

    # Display the extracted risks
    if st.session_state['weight_before'] > 0:
        class_scores = df_policies.groupby('Class')['Weight'].sum()
        class_percentages_df = pd.DataFrame({
            'Class': class_scores.index,
            'Percentage': (class_scores / st.session_state['weight_before'] * 100)
        })
        doughnut_chart = alt.Chart(class_percentages_df).mark_arc(innerRadius=100).encode(
            theta='Percentage:Q',
            color='Class:N',
            tooltip=['Class:N', alt.Tooltip('Percentage:Q', format='.2f')]
        ).properties(
            title='Percentage of Total Score by Class',
            width=400,
            height=400
        )
        st.altair_chart(doughnut_chart, use_container_width=True)
        risk_adjustments = apply_consequence_adjustments(df_risks)

    
    # Toggle to show effects of risk consequences
    show_effects = st.checkbox("Show Effects of Risk Consequences on Score")

    # Display the effects of risk consequences
    if show_effects:
        _, st.session_state['weight_after'] = calculate_and_update_weights(onto_path, risk_adjustments)
        st.metric(label="Adjusted Total Weight After Risk Consequences", value=st.session_state['weight_after'])
        
        percentage_change = ((st.session_state['weight_after'] - st.session_state['weight_before']) / st.session_state['weight_before']) * 100
        st.metric(label="Percentage Change in Weight Due to Risk Consequences", value=f"{percentage_change:.2f}%")
    else:
        st.metric(label="Total Weight (Unadjusted)", value=st.session_state['weight_before'])
   
    # Close the connection
    conn.close()
if __name__ == "__main__":
    app()
