# This script is used to display the vulnerabilities and patches dashboard

# Importing required libraries
from owlready2 import get_ontology
import pandas as pd
import streamlit as st
import altair as alt


# Load the ontology only once during the app's lifecycle.
if 'onto' not in st.session_state:
    onto_path = "/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/csonto-edit.rdf"
    st.session_state['onto'] = get_ontology(onto_path).load()

# Function to get data about vulnerabilities and their patch status
def get_vulnerabilities_and_patches(onto):
    vulnerabilities = onto.search(type = onto.VulnerabilityList)
    vuln_patch_map = {vuln.name: False for vuln in vulnerabilities}  # Initialize all as not patched

    patches = onto.search(type = onto.PatchList)
    for patch in patches:
        for vuln in patch.ProvidePatch:
            vuln_patch_map[vuln.name] = True  # Mark as patched

    vulnerabilities_data = [{
        "ID": vuln.vulnerabilityID[0] if vuln.vulnerabilityID else "N/A",
        "Name": vuln.vulnerabilityName[0] if vuln.vulnerabilityName else "N/A",
        "Description": vuln.vulnerabilityDescription[0] if vuln.vulnerabilityDescription else "N/A",
        "Has Patch": "Yes" if vuln_patch_map[vuln.name] else "No"
    } for vuln in vulnerabilities]

    return pd.DataFrame(vulnerabilities_data)

# Function to get data about patches
def get_patches_data(onto):
    patches = onto.search(type = onto.PatchList)
    patches_data = [{
        "ID": patch.patchID[0] if patch.patchID else "N/A",
        "Name": patch.patchName[0] if patch.patchName else "N/A",
        "Release Date": patch.patchReleaseDate[0] if patch.patchReleaseDate else "N/A"
    } for patch in patches]

    return pd.DataFrame(patches_data)

# Main app function
def app():
    onto = st.session_state['onto']
    VULN_STATUS = 'Has Patch'
    st.title('Vulnerabilities and Patches Dashboard')

    vulnerabilities_df = get_vulnerabilities_and_patches(onto)
    
    # Display overview of vulnerabilities
    st.header("Vulnerabilities Overview")
    st.dataframe(vulnerabilities_df)

    # Visualization: Count of vulnerabilities by patch status
    st.header("Vulnerabilities by Patch Status")
    chart_data = vulnerabilities_df[VULN_STATUS].value_counts().reset_index()
    chart_data.columns = [VULN_STATUS, 'Count']
    c = alt.Chart(chart_data).mark_bar().encode(x=VULN_STATUS, y='Count', color=VULN_STATUS, tooltip=[VULN_STATUS, 'Count']).properties(width=600, height=400)
    st.altair_chart(c, use_container_width=True)

    # Display lists of vulnerabilities with and without patches
    st.header("Detailed Lists")
    st.subheader("Vulnerabilities with Patches")
    st.dataframe(vulnerabilities_df[vulnerabilities_df[VULN_STATUS] == "Yes"])
    st.subheader("Vulnerabilities without Patches")
    st.dataframe(vulnerabilities_df[vulnerabilities_df[VULN_STATUS] == "No"])

if __name__ == "__main__":
    app()
