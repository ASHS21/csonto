from owlready2 import get_ontology
import pandas as pd
import streamlit as st
import altair as alt
import random
from neo4j import GraphDatabase
import json
from py2neo import Graph
from pyvis.network import Network


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

def insert_data_into_neo4j(neo4j_conn, risks, vulnerabilities, policies):
    with neo4j_conn.driver.session() as session:
        # Insert vulnerabilities into Neo4j
        for vuln in vulnerabilities:
            session.run("""
                MERGE (v:Vulnerability {id: $id})
                ON CREATE SET v.name = $name, v.description = $description, v.effects = $effects
                MERGE (s:Score {value: 'Low'})
                MERGE (v)-[:CAUSED_BY]->(s)
                """, 
                id=vuln['vulnerabilityID'], 
                name=vuln['vulnerabilityName'], 
                description=vuln['vulnerabilityDescription'], 
                effects=vuln['vulnerabilityEffects'])

        # Insert risks into Neo4j
        for risk in risks:
            session.run("""
                MERGE (r:Risk {id: $id})
                ON CREATE SET r.name = $name, r.effects = $effects, r.decisions = $decisions
                MERGE (s:Score {value: 'Low'})
                MERGE (r)-[:CAUSED_BY]->(s)
                """, 
                id=risk['CaseID'], 
                name=risk['CaseName'], 
                effects=risk['RiskEffects'], 
                decisions=risk['RiskDecisions'])

        # Insert policies into Neo4j
        # Now, insert the policies
        for policy in policies:
            session.run("""
                MERGE (p:Policy {id: $id})
                ON CREATE SET p.hasWeight = $hasWeight, p.status = $status
                MERGE (s:Score {value: 'Low'})
                MERGE (p)-[:CAUSED_BY]->(s)
                """, 
                id=policy['PolicyID'], 
                hasWeight=policy['HasWeight'], 
                status=policy['Status'])

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
                    # Adjust weight based on risk consequence
                    adjustment_factor = risk_adjustments[instance.name]
                    adjusted_weight = weight * adjustment_factor
                else:
                    # Randomly assign a new status if no risk adjustment is specified
                    new_status = random.choice(["Implemented", "Violated", "Bypassed", "Ignored"])
                    instance.Status = [new_status]
                    adjusted_weight = weight if new_status == "Implemented" else 0

                instance.HasWeight = [adjusted_weight]
                total_weight_after += adjusted_weight

    return total_weight_before, total_weight_after

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

def extract_non_compliant_policies(onto_path):
    onto = get_ontology(onto_path).load()
    
    non_compliant_policies = []
    # Assuming 'Asset_Management_Policies' is the root class for all policies you're interested in
    policy_classes = [
        "http://FYP-ASHS21/csonto#Asset_Management_Policies",
        "http://FYP-ASHS21/csonto#Assurance_Policies",
        "http://FYP-ASHS21/csonto#Change_Management_Policies",
        "http://FYP-ASHS21/csonto#Emerging_Technologies_Policies",
        "http://FYP-ASHS21/csonto#Governance_Policies",
        "http://FYP-ASHS21/csonto#Security_Operations_Policies"
    ]

    for class_iri in policy_classes:
        policy_class = onto.search_one(iri=class_iri)
        if policy_class:
            for policy in policy_class.instances():
                status = getattr(policy, 'Status', [None])[0]
                if status in ["Violated", "Ignored", "Bypassed"]:
                    non_compliant_policies.append({
                        'PolicyID': policy.name,
                        'HasWeight': getattr(policy, 'HasWeight', [None])[0],
                        'Status': status,
                        # Extract other relevant properties as needed
                    })
    
    return non_compliant_policies

def create_vis_network(neo4j_conn, query):
    # Create a new PyVis network
    net = Network(height="100%", width="100%", bgcolor="#222222", font_color="white")
    
    # Connect to Neo4j using py2neo's session
    with neo4j_conn.driver.session() as session:
        # Run the Cypher query
        results = session.run(query)
        
    # Turn off physics for static graph
    net.toggle_physics(False)
    
    # Generate the HTML and Javascript for rendering
    try:
        net.show('graph.html')
    except Exception as e:
        print(e)
    
    # Return HTML file path for display in Streamlit
    return 'graph.html'
    
 

def app():

    config_path = '/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/pages/config.json'  
    
    # Initialize connection
    conn = Neo4jConnection(config_path)
    st.title('Policy and Risk Management Dashboard')
    onto_path = "/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/csonto-edit.rdf"
    
    # Extract information
    vulnerabilities_without_patches = extract_unpatched_vulnerabilities(onto_path)
    critical_accepted_risks = extract_critical_accepted_risks(onto_path)
    extracted_policies = extract_non_compliant_policies(onto_path)

    # Display extracted information in the app for verification
    st.subheader("Unpatched Vulnerabilities")
    st.write(vulnerabilities_without_patches)

    st.subheader("Critical Accepted Risks")
    st.write(critical_accepted_risks)

    
   

    # If a button is clicked, insert the data into Neo4j
    if st.button('Update Neo4j Graph'):
        insert_data_into_neo4j(conn, critical_accepted_risks, vulnerabilities_without_patches, extracted_policies)
        st.success("Neo4j Graph updated with new vulnerabilities, risks, and policies!")

    if 'weight_before' not in st.session_state:
        st.session_state['weight_before'], _ = calculate_and_update_weights(onto_path)
        st.session_state['weight_after'] = st.session_state['weight_before']

    st.metric(label="Initial Total Weight Across All Classes", value=st.session_state['weight_before'])

    if st.button('Recalculate Weights'):
        _, st.session_state['weight_after'] = calculate_and_update_weights(onto_path)
        st.metric(label="Total Weight Across All Classes After Changes", value=st.session_state['weight_after'])
        if st.session_state['weight_before'] > 0:
            percentage_change = (st.session_state['weight_after'] / st.session_state['weight_before']) * 100
            st.metric(label="Percentage of Initial Total Weight", value=f"{percentage_change:.2f}%")
        else:
            st.error("Initial total weight is zero, cannot calculate percentage.")

    df_policies = extract_policy_data(onto_path)
    df_risks = extract_risk_information(onto_path)

    status_counts = df_policies['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Counts']
    chart1 = alt.Chart(status_counts).mark_bar().encode(
        x='Status',
        y='Counts',
        color='Status:N',
        tooltip=['Status', 'Counts']
    ).properties(title='Policy Counts by Status')
    st.altair_chart(chart1, use_container_width=True)

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

    if show_effects:
        _, st.session_state['weight_after'] = calculate_and_update_weights(onto_path, risk_adjustments)
        st.metric(label="Adjusted Total Weight After Risk Consequences", value=st.session_state['weight_after'])
        
        percentage_change = ((st.session_state['weight_after'] - st.session_state['weight_before']) / st.session_state['weight_before']) * 100
        st.metric(label="Percentage Change in Weight Due to Risk Consequences", value=f"{percentage_change:.2f}%")
    else:
        st.metric(label="Total Weight (Unadjusted)", value=st.session_state['weight_before'])
   
    conn.close()
if __name__ == "__main__":
    app()
