import streamlit as st
from owlready2 import get_ontology
import pandas as pd
from geopy.geocoders import Nominatim
import pydeck as pdk
import numpy as np
import matplotlib.pyplot as plt
import altair as alt


# Initialize Nominatim Geocoder
geolocator = Nominatim(user_agent="streamlit-app")

# Streamlit page configuration
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Function to geocode location names into latitude and longitude
@st.cache_resource
def geocode_location(location, attempt=1, max_attempts=3):
    try:
        location_obj = geolocator.geocode(location, timeout=10)
        return (location_obj.latitude, location_obj.longitude) if location_obj else (None, None)
    except Exception as e:
        if attempt < max_attempts:
            return geocode_location(location, attempt + 1, max_attempts)
        st.error(f"Geocoding error for {location}: {e}")
        return None, None
    
@st.cache(allow_output_mutation=True)
def load_assets_data():
    onto_path = "/Users/alialmoharif/Desktop/FYP/Code/final-year-project-ASHS21/csonto/target/csonto/dashboards/csonto-edit.rdf" 
    onto = get_ontology(onto_path).load()
    AssetsList = onto.search_one(iri="http://FYP-ASHS21/csonto#AssetsList")

    if not AssetsList:
        raise ValueError("AssetsList class not found in the ontology")

    assets_data = []
    for asset_instance in AssetsList.instances():
        last_known_location = getattr(asset_instance, 'lastKnownLocation', None)
        original_place = getattr(asset_instance, 'originalPlace', None)

        last_known_location = str(last_known_location) if last_known_location else None
        original_place = str(original_place) if original_place else None
        
        # Geocode locations to get latitude and longitude
        last_known_lat_lon = geocode_location(last_known_location) if last_known_location else (None, None)
        original_place_lat_lon = geocode_location(original_place) if original_place else (None, None)
        
        # Include last known location data if available
        if last_known_lat_lon != (None, None):
            assets_data.append({
                'Name': asset_instance.name,
                'TypeOfAsset': getattr(asset_instance, 'typeOfAsset', None),
                'AssetID': getattr(asset_instance, 'AssetID', None),
                'LocationType': 'LastKnownLocation',
                'City': last_known_location,  # Storing the city name
                'Lat': last_known_lat_lon[0],
                'Lon': last_known_lat_lon[1],
                'Color': np.random.randint(0, 256, 4).tolist(),
            })
        
        # Include original place data if available
        if original_place_lat_lon != (None, None):
            assets_data.append({
                'Name': asset_instance.name,
                'TypeOfAsset': getattr(asset_instance, 'typeOfAsset', None),
                'AssetID': getattr(asset_instance, 'AssetID', None),
                'LocationType': 'OriginalPlace',
                'City': original_place,  # Storing the city name
                'Lat': original_place_lat_lon[0],
                'Lon': original_place_lat_lon[1],
                'Color': np.random.randint(0, 256, 4).tolist(),
            })

    return pd.DataFrame(assets_data)


def generate_location_bar_chart(df, title):
    """Generates and displays a bar chart for asset counts by city."""
    counts = df['City'].value_counts().reset_index()
    counts.columns = ['City', 'NumberOfAssets']
    st.write(f"### {title}")
    st.bar_chart(counts.set_index('City'))

def generate_asset_type_distribution(df, unique_key):
    """Generates and displays the distribution of asset types for a selected city."""
    # Ensure 'TypeOfAsset' is converted to a string to handle non-hashable types
    df['TypeOfAsset'] = df['TypeOfAsset'].apply(lambda x: ', '.join(str(item) for item in x) if isinstance(x, list) else str(x))
    
    location_selection = st.selectbox('Select a city to view asset type distribution:', df['City'].unique(), key=unique_key)
    
    if location_selection:
        filtered_data = df[df['City'] == location_selection]
        type_counts = filtered_data.groupby('TypeOfAsset').size().reset_index(name='Count')
        st.write(f"### Asset Type Distribution in {location_selection}")
        st.bar_chart(type_counts.set_index('TypeOfAsset'))


# Main function to render the Streamlit app
def app():
    st.title("Assets Location Map")
    
    # Initialize session state for search if not already done
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False
    
    # Add a search bar and get the input from the user
    search_id = st.text_input("Enter Asset ID to search", "")
    
    # Load assets data from the ontology
    assets_df = load_assets_data()
    
    # "Search" button
    if st.button("Search"):
        st.session_state.search_performed = True
    
    # If the search ID is provided and the "Search" button has been clicked
    if search_id and st.session_state.search_performed:
        filtered_df = assets_df[assets_df['AssetID'] == search_id]
        if not filtered_df.empty:
            # Display the search results
            st.write(f"Search results for Asset ID: {search_id}")
            st.dataframe(filtered_df)
        else:
            st.warning(f"No assets found with ID: {search_id}")
        
        # "Reset Search" button to clear the search and refresh the app
        if st.button("Reset Search"):
            st.session_state.search_performed = False
            st.experimental_rerun()
    
    # If no search has been performed or after resetting the search, show default view
    if not st.session_state.search_performed:
        if not assets_df.empty:
            # Generate pairs of locations for the LineLayer
            line_data = []
            for _, group in assets_df.groupby('AssetID'):
                if group.shape[0] > 1:  # Ensure there are at least two points (original and last known location)
                    sorted_group = group.sort_values(by='LocationType', ascending=False)
                    line_record = {
                        'coordinates': [
                            [sorted_group.iloc[0]['Lon'], sorted_group.iloc[0]['Lat']],
                            [sorted_group.iloc[1]['Lon'], sorted_group.iloc[1]['Lat']]
                        ],
                        'color': sorted_group.iloc[0]['Color']  # Use the color of the first record
                    }
                    line_data.append(line_record)

            # Creating ScatterplotLayer for assets
            scatterplot_layers = pdk.Layer(
                "ScatterplotLayer",
                assets_df,
                get_position=['Lon', 'Lat'],
                get_color='Color',
                get_radius=100,
                pickable=True,
                tooltip={"html": "{LocationType}<br><b>{Name}</b><br>Type: {TypeOfAsset}<br>Asset ID: {AssetID}"},
            )

            # Creating LineLayer for asset movements
            line_layer = pdk.Layer(
                "LineLayer",
                line_data,
                get_source_position='coordinates[0]',
                get_target_position='coordinates[1]',
                get_color='color',
                get_width=5,
                pickable=True,
                tooltip="Movement path"
            )

            # Define initial view state for the map
            view_state = pdk.ViewState(
                latitude=assets_df['Lat'].mean(),
                longitude=assets_df['Lon'].mean(),
                zoom=2,
                pitch=0,
            )

            # Render the map with the defined layers
            st.pydeck_chart(pdk.Deck(layers=[scatterplot_layers, line_layer], initial_view_state=view_state))

            # Visualizing Number of Assets in Last Known Locations and Original Places
            generate_location_bar_chart(assets_df, "Number of Assets by City")
            generate_asset_type_distribution(assets_df, 'city_dist')
        else:
            st.warning("No asset data loaded or geocoded successfully.")

if __name__ == "__main__":
    app()

