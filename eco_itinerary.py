import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from multiple datasources
attractions_data = pd.read_csv('attractions_name.csv')
emissions_data = pd.read_csv('emissions.csv')

# Merge the emissions data with the attractions data
attractions_data = attractions_data.merge(
    emissions_data, on='attraction_id', how='left')
print(attractions_data.head())
print(emissions_data.head())
print(attractions_data.columns)
attractions_data.isnull().sum()
# Define a function to generate a recommended itinerary based on user preferences


def generate_itinerary(user_preferences):
    # Filter attractions by user preferences
    filtered_attractions = attractions_data.loc[attractions_data['type'].isin(
        user_preferences.columns)]
    print(filtered_attractions.columns)
    # Calculate cosine similarity between user preferences and filtered attractions
    preferences_array = np.array([user_preferences[attraction_type].values[0]
                                 for attraction_type in user_preferences.columns]).reshape(1, -1)
    print(user_preferences.columns)
    similarity_scores = cosine_similarity(
        filtered_attractions[user_preferences.columns].fillna(0), preferences_array)

    # Sort attractions by emissions and similarity scores
    filtered_attractions['score'] = similarity_scores.squeeze()
    sorted_attractions = filtered_attractions.sort_values(
        by=['emissions', 'score'], ascending=[True, False])

    # Get the top recommended attractions that match user preferences
    recommended_attractions = []
    for attraction_type in user_preferences.columns:
        attractions_subset = sorted_attractions[sorted_attractions['type']
                                                == attraction_type][:3]
        attractions_subset = attractions_subset[attractions_subset['score'] > 0.2]
        recommended_attractions.extend(attractions_subset.index)

    # Get the emissions for the recommended attractions
    emissions = sorted_attractions.loc[recommended_attractions, 'emissions'].sum(
    )

    # Return the recommended attractions and emissions as a tuple
    return sorted_attractions.loc[recommended_attractions, ['name', 'type', 'emissions']], emissions


# Define the Streamlit app


def app():
    st.title('Tourist Itinerary Generator')

    # Define the user preferences form
    st.sidebar.title('Select Your Preferences')
    preferences = {}
    for attraction_type in attractions_data['type'].unique():
        preferences[attraction_type] = st.sidebar.slider(
            attraction_type, 0, 10, 5)

    # Convert user preferences to a dataframe
    user_preferences = pd.DataFrame([preferences])

    # Generate recommended itinerary based on user preferences
    itinerary, emissions = generate_itinerary(user_preferences)

    # Display the recommended itinerary
    st.write('## Your Recommended Itinerary')
    selected_attractions = st.multiselect(
        'Select Attractions to Remove or Change', itinerary['name'])
    if len(selected_attractions) > 0:
        itinerary = itinerary[~itinerary['name'].isin(selected_attractions)]
    st.write(itinerary)

    # Display the total greenhouse gas emissions
    st.write('## Total Greenhouse Gas Emissions')
    st.write(emissions)


if __name__ == '__main__':
    app()
