import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from PIL import Image
import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import pickle
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt 
from streamlit_option_menu import option_menu

# Load your scaler and linear regression models for each target column
scaler = joblib.load("last_scaler.pkl")
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear', 'Communication', 'Education', 'Food and non-alcoholic beverages', 'Headline_CPI', 'Health', 'Household contents and services', 'Housing and utilities', 'Miscellaneous goods and services', 'Recreation and culture', 'Restaurants and hotels ', 'Transport']
model_dict = {target_col: joblib.load(f"{target_col}_model.pkl") for target_col in target_cols}


# Resize all images to a consistent size
def resize_image(image_path, size=(150, 150)):
    image = Image.open(image_path)
    image = image.resize(size)
    return image
# Set page configuration and title
st.set_page_config(page_title="CPI Vision Prediction", layout="wide")

# Load the dataset
input_data = pd.read_csv('train.csv')

# Sidebar
with st.sidebar:
    image1 = Image.open('CPI-logo.png')
    st.image(image1, caption='CPI Vision Prediction')
    page_selection = option_menu(
        menu_title=None,
        options=["Overview", "Prediction", "Dashboard", "Meet the Team", "Contact Us"],
        icons=['file-earmark-text', 'eye', 'graph-up', 'file-earmark-spreadsheet', 'envelope'],
        menu_icon='cast',
        default_index=0,
        styles={"container": {'padding': '0!important', 'background_color': 'red'},
                'icon': {'color': 'red', 'font-size': '18px'},
                'nav-link': {
                    'font-size': '15px',
                    'text-align': 'left',
                    'margin': '0px',
                    '--hover-color': '#4BAAFF',
                },
                'nav-link-selected': {'background-color': '#6187D2'},
                }
    )
    


# Add future months and years to the dataset
current_date = datetime.date(2023, 4, 30)  # Starting from April 2023
end_date = datetime.date(2025, 12, 30)  # Extend data up to December 2024

while current_date <= end_date:
    year_month = current_date.strftime('%Y-%m')
    month = current_date.strftime('%Y-%m-%d')
    input_data = input_data.append({'year_month': year_month, 'Month': month}, ignore_index=True)
    current_date = current_date + pd.DateOffset(months=1)

# Load your CPI dataset (replace 'Book6.csv' with your dataset file path)
def load_data():
    data = pd.read_csv('Book6.csv')
    return data

cpi_data = load_data()

# Create a dictionary to store previous month's predictions for each category
previous_month_predictions = {category: None for category in target_cols}

# Define a function to display the "Meet the Team" page
def meet_the_team():
    st.title("Meet the Team")
    
    # Add team members with their pictures and descriptions
    team_members = [
        {"name": "Sibongile Mokoena", "position": "Junior Data Scientist", "image": "Sibongile.jpeg", "description": "Sibongile is a data scientist with expertise in machine learning and data analysis."},
        {"name": "Manoko Langa", "position": "Web Developer", "image": "manoko.jpeg", "description": "Manoko is a web developer responsible for creating the Streamlit app."},
        {"name": "Zandile Mdiniso", "position": "Data Scientist", "image": "zand.jpeg", "description": "Similar to Manoko, Zandile is a data scientist with expertise in data analysis and machine learning."},
        {"name": "Thando Vilakazi", "position": "Business Analyst", "image": "thando.jpeg", "description": "Thando is a business analyst responsible for the valuable insights extracted in this project."},
        {"name": "Zweli Khumalo", "position": "Business Analyst", "image": "zweli.jpeg", "description": "Zweli is a business analyst responsible for the valuable insights extracted in this project."},
    
    ]
    
    # Create columns for images and descriptions
    columns = st.columns(len(team_members))
    for i, member in enumerate(team_members):
        with columns[i]:
            st.image(resize_image(member['image']), caption=member['name'], use_column_width=True)
            st.write(f"**{member['name']}**")
            st.write(f"**Position**: {member['position']}")
            st.write(member['description'])
# Page selection
if page_selection == "Overview":
    
    st.title(" CPI Vision Application Overview")
    # Add the "Overview" page
     # Resize and display the image
     # Create a container to center-align the content
    #center_container = st.beta_container()

    # # Resize and display the image within the container
    # with center_container:
    #     st.image('CPI-basket.png', width=500) 
    # image = Image.open('CPI-basket.png')
    # resized_image = image.resize((500, 300))  # A

    # Resize and display the image, centered
    #image = Image.open('CPI-basket.png')
    #resized_image = image.resize((400, 250))  # Adjust the size as needed
    #st.title("CPI Overview")
    # Center-align the image
    #st.image(resized_image, use_column_width=True, caption='CPI Overview')

    # Introduction
    st.write("Welcome to the CPI (Consumer Price Index) Overview page.")
    st.write("This page provides a general overview of CPI data and the purpose of this application.")
    
    # What is CPI
    st.header("What is CPI?")
    st.write("The Consumer Price Index (CPI) is a measure of the average change over time in the prices paid by urban consumers "
             "for a market basket of consumer goods and services. It is a crucial indicator of inflation and economic stability.")
    
    # Purpose of the App
    st.header("Purpose of the CPI Vision App")
    st.write("The CPI Vision App is designed to assist in predicting future CPI values for various categories based on historical data. "
             "Users can select a category, a month, and a year to get predictions.")
    
    # Data Source
    #st.header("Data Source")
    st.write("The data used in this application is sourced from the CPI Nowcast Challenge, and it covers the period from January 2022 to March 2023.")
    
    

elif page_selection == "Prediction":
    st.title("CPI Vision App")
    #image = Image.open("CPI-logo.png")
    #st.image(image, caption="CPI Logo")
    st.header("Predict CPI")
    
    # User input - Select Category
    category = st.selectbox("Select Category", target_cols)

    # User input - Select Month and Year
    selected_month = st.selectbox("Select Month", ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"])
    selected_year = st.selectbox("Select Year", ["2023", "2024", "2025"])

    # Create a function to preprocess input data for prediction based on the selected month and year
    def preprocess_input_data(selected_month, selected_year):
        global input_data  # Declare input_data as a global variable
        
        feats_to_lag = [col for col in input_data.columns if col not in ['Month', 'year_month']]
        for col in feats_to_lag:
            for i in range(1, 12):
                input_data[f"prev_{i}_month_{col}"] = input_data[col].shift(i)
                    
        # Fill null values in lag columns with random values, sampled from the same column
        lag_columns = [col for col in input_data.columns if col.startswith("prev_")]
        for col in lag_columns:
            null_mask = input_data[col].isnull()
            num_nulls = null_mask.sum()
            if num_nulls > 0:
                random_values = np.random.choice(input_data[col].dropna(), num_nulls)
                input_data.loc[null_mask, col] = random_values

        input_data = input_data.drop(columns=target_cols + ['Total_Local Sales', 'Total_Export_Sales'])
        
        selected_date = f"{selected_year}-{selected_month}"
        selected_data = input_data[input_data['year_month'] == selected_date]

        return selected_data

    # Add a button to trigger predictions
    if st.button("Predict CPI"):
        
        input_data = preprocess_input_data(selected_month, selected_year)

        if not input_data.empty:
            input_scaled = scaler.transform(input_data.drop(columns=['Month', 'year_month']))

            lr_model = model_dict[category]
            predicted_cpi = lr_model.predict(input_scaled)
            
            # Display the predicted CPI value to the user with larger text using HTML styling
            st.markdown(f"<h2>Predicted CPI for {category} in {selected_year}-{selected_month}: {predicted_cpi[0]:.2f}</h2>", unsafe_allow_html=True)

            prev_month_key = category  # Use the category name as the key

            # Fetch previous month's prediction from the actual data
            prev_month_col = f"prev_1_month_{category}"
            prev_month = input_data.loc[input_data['year_month'] == f"{selected_year}-{selected_month}"][prev_month_col].values[0]

            # Update the dictionary with the previous month's prediction
            previous_month_predictions[prev_month_key] = prev_month

            prev_month_data = pd.DataFrame({
                'Month': ['Previous Month', 'Current Month'],
                'CPI Value': [prev_month, predicted_cpi[0]]
            })

            percentage_change = ((predicted_cpi[0] - prev_month) / prev_month) * 100

            if percentage_change > 0:
                change_icon = "📈"
                change_text = f"Increased by {percentage_change:.2f}%"
            elif percentage_change < 0:
                change_icon = "📉"
                change_text = f"Decreased by {abs(percentage_change):.2f}%"
            else:
                change_icon = "📊"
                change_text = "No change"

            # Display the change card/legend with larger text using HTML styling
            st.markdown(f"<h3>Change: {change_icon} {change_text}</h3>", unsafe_allow_html=True)

            fig, ax = plt.subplots()
            ax.bar(prev_month_data['Month'], prev_month_data['CPI Value'], color=['red', 'blue'])
            ax.set_xlabel('Month')
            ax.set_ylabel('CPI Value')
            ax.set_title(f'{category} CPI Comparison')

            for i, v in enumerate(prev_month_data['CPI Value']):
                ax.text(i, v, f'{v:.2f}', va='bottom', ha='center', fontsize=12)

            ax.tick_params(axis='x', which='both', bottom=False)
            ax.set_facecolor('#F5F5F5')

            st.pyplot(fig, use_container_width=True)
        
        else:
            st.write(f"No data available for {selected_year}-{selected_month}. Please select a different month and year.")

# elif page_selection == "Dashboard":
#     st.header("Dashboard")
    
#     # User input - Select Category for the dashboard
#     selected_category = st.selectbox("Select Category for the Dashboard", target_cols)
    
#     # Filter data for the selected category
#     category_data = input_data[['year_month', selected_category]].copy()
    
#     # Create a column for percentage change
#     category_data['Percentage Change'] = (category_data[selected_category] - category_data[selected_category].shift(1)) / category_data[selected_category].shift(1) * 100
    
#     # Display the selected category name
#     st.write(f"Dashboard for: {selected_category}")
    
#     # Create a subplot with two traces (bar and line)
#     fig = make_subplots(specs=[[{"secondary_y": True}]])
    
#     # Add bar trace (CPI Values)
#     fig.add_trace(
#         go.Bar(x=category_data['year_month'], y=category_data[selected_category], name='CPI Values', marker_color='blue'),
#         secondary_y=False,
#     )
    
#     # Add line trace (Percentage Change)
#     fig.add_trace(
#         go.Scatter(x=category_data['year_month'], y=category_data['Percentage Change'], mode='lines+markers', name='Percentage Change', line=dict(color='red')),
#         secondary_y=True,
#     )
    
#     # Update layout
#     fig.update_layout(
#         title=f'{selected_category} CPI and Percentage Change',
#         xaxis_title="Month",
#         yaxis_title="CPI Values",
#         yaxis2_title="Percentage Change",
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#     )

#     # Set Y-axis limits for the visual under the dashboard page
#     fig.update_yaxes(range=[97, 105], secondary_y=False)

#     st.plotly_chart(fig, use_container_width=True)

#     # Calculate the percentage contribution of each category in the CPI value
#     contribution_data = input_data.copy()
#     contribution_data['Year'], contribution_data['Month'] = contribution_data['year_month'].str.split('-', 1).str
#     contribution_data = contribution_data.groupby(['Year', 'Month'])[target_cols].mean()
#     contribution_data = (contribution_data / contribution_data.sum(axis=1, skipna=True).values.reshape(-1, 1)) * 100
    
#     # Create a treemap chart for category contributions
#     treemap_fig = go.Figure(go.Treemap(
#         labels=[f"{year}-{month}" for year, month in contribution_data.index],
#         parents=['' for _ in contribution_data.index],
#         values=contribution_data[selected_category].values,
#     ))

#     # Customize the treemap layout
#     treemap_fig.update_layout(
#         title=f'{selected_category} Contribution to CPI Over Time',
#     )

#     # Display the treemap chart
#     st.plotly_chart(treemap_fig, use_container_width=True)

elif page_selection == "Dashboard":
    st.header("Dashboard Insights")

    # Add a year slicer
    selected_year = st.slider('Select a Year', min_value=int(cpi_data['Year'].min()), max_value=int(cpi_data['Year'].max()))

    # Add a category selector
    selected_category = st.selectbox('Select a Category', cpi_data['Category'].unique())

    # Filter the data based on the selected year and category
    filtered_data = cpi_data[(cpi_data['Year'] == selected_year) & (cpi_data['Category'] == selected_category)]

    # Convert 'MONTH' column to datetime format
    filtered_data['MONTH'] = pd.to_datetime(filtered_data['MONTH'], format='%B')

    # Sort the filtered data by 'MONTH' column
    filtered_data = filtered_data.sort_values(by='MONTH')

    # Extract the month names for x-axis labels
    month_names = filtered_data['MONTH'].dt.strftime('%B')

    # Create a subplot with two traces (bar and line)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bar trace (CPI Values)
    fig.add_trace(
        go.Bar(x=month_names, y=filtered_data['Value'], name='Value', marker_color='blue'),
        secondary_y=False,
    )

    # Add line trace (Percentage Change)
    fig.add_trace(
        go.Scatter(x=month_names, y=filtered_data['Percentage Change (From Prior Month)'], mode='lines+markers', name='Percentage Change (From Prior Month)', line=dict(color='red')),
        secondary_y=True,
    )

    # Update layout
    fig.update_layout(
        title='CPI and Percentage Change',
        xaxis_title="Month",
        yaxis_title="CPI Values",
        yaxis2_title="Percentage Change",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x',  # Enable hover mode for the entire figure
    )

    # Set Y-axis limits for the visual
    fig.update_yaxes(range=[97, 105], secondary_y=False)

    # Streamlit code to display the plot
    st.subheader('CPI and Percentage Change Visualization')
    st.plotly_chart(fig, use_container_width=True)

elif page_selection == "Meet the Team":
    meet_the_team()
    

elif page_selection == "Contact Us":
        st.title('Contact Us!')
        st.markdown("Have a question or want to get in touch with us? Please fill out the form below with your email "
                    "address, and we'll get back to you as soon as possible. We value your privacy and assure you "
                    "that your information will be kept confidential.")
        st.markdown("By submitting this form, you consent to receiving email communications from us regarding your "
                    "inquiry. We may use the email address you provide to respond to your message and provide any "
                    "necessary assistance or information.")
        with st.form("Email Form"):
            subject = st.text_input(label='Subject', placeholder='Please enter subject of your email')
            fullname = st.text_input(label='Full Name', placeholder='Please enter your full name')
            email = st.text_input(label='Email Address', placeholder='Please enter your email address')
            text = st.text_area(label='Email Text', placeholder='Please enter your text here')
            uploaded_file = st.file_uploader("Attachment")
            #submit_res = st.form_submit_button("Send")
        st.markdown("Thank you for reaching out to us. We appreciate your interest in our loan approval web "
                    "application and look forward to connecting with you soon")