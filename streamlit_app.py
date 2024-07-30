import streamlit as st
import pandas as pd
import math
from pathlib import Path
from huggingface_hub import from_pretrained_keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP and Air Quality Dashboard',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

@st.cache_resource
def load_aqi_model():
    """Load the pretrained AQI model."""
    return from_pretrained_keras("keras-io/timeseries_forecasting_for_weather")

@st.cache_data
def get_aqi_data():
    """Load and preprocess AQI data."""
    data = pd.read_csv('/content/daily_aqi_by_county_2020.csv')
    data = data[['State Name', 'county Name', 'Date', 'AQI']]
    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date']).astype(int) / 10**9
    data = data.sample(frac=0.10, random_state=42)
    
    scaler = MinMaxScaler()
    data[['Date', 'AQI']] = scaler.fit_transform(data[['Date', 'AQI']])
    
    state_encoder = LabelEncoder()
    county_encoder = LabelEncoder()
    data['State Name'] = state_encoder.fit_transform(data['State Name'])
    data['county Name'] = county_encoder.fit_transform(data['county Name'])
    
    return data, scaler

def prepare_aqi_sequences(data):
    """Prepare AQI data sequences for prediction."""
    features = data[['State Name', 'county Name', 'Date']].values
    num_features = 7
    sequence_length = 120
    
    features = np.pad(features, ((0, 0), (0, num_features - features.shape[1])), 'constant')
    
    num_samples = len(features)
    if num_samples < sequence_length:
        st.warning(f"Not enough data to form sequences of length {sequence_length}")
        return None
    
    sequences = []
    for i in range(num_samples - sequence_length + 1):
        sequences.append(features[i:i+sequence_length])
    return np.array(sequences)

# -----------------------------------------------------------------------------
# Load data and model

gdp_df = get_gdp_data()
aqi_model = load_aqi_model()
aqi_data, aqi_scaler = get_aqi_data()

# -----------------------------------------------------------------------------
# Draw the actual page

st.title('GDP and Air Quality Dashboard')

st.markdown('''
Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website and explore Air Quality predictions.
''')

# GDP Section
st.header('GDP Data', divider='gray')

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

if not selected_countries:
    st.warning("Select at least one country")

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.subheader('GDP over time')
st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

st.subheader(f'GDP in {to_year}')

cols = st.columns(4)

first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )

# Air Quality Section
st.header('Air Quality Predictions', divider='gray')

if st.button('Generate AQI Predictions'):
    sequences = prepare_aqi_sequences(aqi_data)
    
    if sequences is not None:
        predictions = aqi_model.predict(sequences)
        predictions = predictions.squeeze().reshape(-1, 1)
        aqi_min = aqi_scaler.data_min_[1]
        aqi_max = aqi_scaler.data_max_[1]
        predicted_aqi = predictions * (aqi_max - aqi_min) + aqi_min
        
        st.subheader('Predicted AQI Statistics')
        st.write(f"Mean: {np.mean(predicted_aqi):.2f}")
        st.write(f"Min: {np.min(predicted_aqi):.2f}")
        st.write(f"Max: {np.max(predicted_aqi):.2f}")
        
        st.subheader('Sample Predictions')
        for i in range(10):
            st.write(f"Predicted AQI for sequence {i+1}: {predicted_aqi[i][0]:.2f}")
        
        st.subheader('AQI Prediction Distribution')
        st.histogram_chart(pd.DataFrame(predicted_aqi, columns=['Predicted AQI']))
    else:
        st.error("Unable to generate predictions due to insufficient data.")