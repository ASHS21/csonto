from owlready2 import get_ontology
import pandas as pd
import streamlit as st
import altair as alt
import random

def calculate_and_update_weights(onto_path):
    onto = get_ontology(onto_path).load()

    classes_iris = [
        "http://FYP-ASHS21/csonto/Asset_Management_Policies",
        "http://FYP-ASHS21/csonto/Assurance_Policies",
        "http://FYP-ASHS21/csonto/Change_Management_Policies",
        "http://FYP-ASHS21/csonto/Emerging_Technologies_Policies",
        "http://FYP-ASHS21/csonto/Governance_Policies",
        "http://FYP-ASHS21/csonto/Security_Operations_Policies"
    ]

    # Initialize total weights
    total_weight_before, total_weight_after = 0, 0

    # Iterate over each class and its instances
    for class_iri in classes_iris:
        policy_class = onto.search_one(iri=class_iri)
        if policy_class:
            for instance in policy_class.instances():
                # Assume 'HasWeight' is always defined for each instance, defaulting to [0] if not
                weight = getattr(instance, "HasWeight", [0])[0]
                total_weight_before += weight  # Sum weights before any changes

                # Randomly assign a new status
                new_status = random.choice(["Implemented", "Violated", "Bypassed", "Ignored"])
                instance.Status = [new_status]
                # Set 'HasWeight' to [0] for "Violated", "Bypassed", "Ignored"; retain original weight for "Implemented"
                instance.HasWeight = [0] if new_status in ["Violated", "Bypassed", "Ignored"] else [weight]
                # Update 'total_weight_after' based on the potentially updated 'HasWeight'
                total_weight_after += instance.HasWeight[0]

    # Return the total weights before and after the changes
    # Removed the division by 'total_policies' as instructed
    return total_weight_before, total_weight_after


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


def app():
    st.title('Policy Dashboard')
    onto_path = "/Users/alisami/Desktop/app/csonto-edit.rdf"

    if 'weight_before' not in st.session_state:
        # Only calculate once per session unless manually refreshed
        st.session_state['weight_before'], _ = calculate_and_update_weights(onto_path)

    weight_before = st.session_state['weight_before']
    
    df = extract_policy_data(onto_path)


    # Visualization 1: Policy Counts by Status
    status_counts = df['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Counts']
    chart1 = alt.Chart(status_counts).mark_bar().encode(
        x=alt.X('Status', sort=None),
        y='Counts',
        color='Status:N',
        tooltip=['Status', 'Counts']
    ).properties(title='Policy Counts by Status')
    st.altair_chart(chart1, use_container_width=True)

    if weight_before > 0:
        # Calculate class percentages based on the total weight before
        class_scores = df.groupby('Class')['Weight'].sum()
        class_percentages_df = pd.DataFrame({
            'Class': class_scores.index,
            'Percentage': (class_scores / weight_before * 100)
        })

        doughnut_chart = alt.Chart(class_percentages_df).mark_arc(innerRadius=100).encode(
            theta=alt.Theta(field="Percentage", type="quantitative"),
            color=alt.Color(field="Class", type="nominal"),
            tooltip=[alt.Tooltip(field="Class", type="nominal"), alt.Tooltip(field="Percentage", type="quantitative", format=".2f")]
        ).properties(
            title='Percentage of Total Score by Class',
            width=400,  # Adjust width and height as needed
            height=400
        )
        
        st.altair_chart(doughnut_chart, use_container_width=True)
    else:
        st.write("Unable to calculate percentages without a valid 'weight_before' value.")

if __name__ == "__main__":
    app()
