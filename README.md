# CPI-VISION
CPI Vision utilizes advanced statistical algorithms and machine learning techniques to analyze extensive datasets of past CPI values, economic indicators, and market trends. By processing this vast amount of information, the tool generates accurate predictions for future CPI values across different categories, such as food and housing.

## Features

- Upload CPI data from a PDF document.
- Select specific categories for CPI prediction.
- Provide input data such as vehicle sales and currency exchange rates.
- Predict CPI values for the selected categories for future months.
- Display previous CPI values for selected categories.

## Prerequisites

Before running the application, make sure you have the following installed:

- Python 3.9 or higher
- Required Python packages (install using `pip`): `streamlit`, `pdfplumber`, `numpy`, `tensorflow`, `scikit-learn`

## Getting Started

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/cpi-prediction.git


# 4. Run the Streamlit application:

   ```bash
   streamlit run CPI_App4.py
   ```

# 5. Upload a CPI PDF document, select categories, provide input data, and click the "Predict CPI" button to see the predictions.

## Usage

1. **Upload CPI PDF Document**: Click the "Upload a CPI PDF document" button to select a PDF file containing CPI data.

2. **Select Categories**: Use the multiselect widget to choose one or more categories for CPI prediction.

3. **Enter Input Data**: Provide input data such as total local sales, total export sales, USD to ZAR exchange rate, GBP to ZAR exchange rate, and EUR to ZAR exchange rate.

4. **Select Prediction Month**: Choose whether you want to predict CPI for the next month, two months later, or three months later.

5. **Predict CPI**: Click the "Predict CPI" button to calculate and display the predicted CPI values for the selected categories.

6. **Previous CPI Values**: After selecting categories, the tool will display the previous CPI values for those categories.

## Models and Data

The tool uses pre-trained deep neural network models for CPI prediction. Models are saved in the format `"{category}_Deep Neural Network_month_{n}.h5"`.

