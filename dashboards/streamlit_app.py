# The script below is the main script for the Streamlit app that displays the different dashboards.

# Import the required libraries
import streamlit as st

# Function to render the home page
def page_home():
    st.title("Welcome to Our Cybersecurity Dashboard")
    st.write("Navigate using the sidebar to access different dashboards.")

# Dynamically import pages to avoid circular imports and reduce startup time
def load_page(page_module):
    """Dynamically import and return the page function from a module."""
    module = __import__(f"pages.{page_module}", fromlist=['app'])
    return module.app

# Main function to render the app
def main():
    st.sidebar.title("Navigation")

    # Dictionary of pages
    pages_dict = {
        "Home": page_home,
        "Score Dashboard": "score_dashboard",
        "Asset Dashboard": "asset_dashboard",
        "Vulnerabilities Dashboard": "vulnerability_dashboard",
    }

    # Sidebar selection
    selection = st.sidebar.selectbox("Choose a page:", list(pages_dict.keys()))

    # Render the selected page
    if selection == "Home":
        page_home()
    else:
        page_func = load_page(pages_dict[selection])
        page_func()

if __name__ == "__main__":
    main()
