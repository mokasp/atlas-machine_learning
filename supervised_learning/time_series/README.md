# Bitcoin Price Time Series Forecasting
This project provides a pipeline to clean, preprocess, and forecast Bitcoin closing prices using an LSTM model.

## Requirements
- Python 3
- pandas
- numpy
- matplotlib
- tensorflow
  
## Dataset
The dataset includes Bitcoin prices at 60-second intervals from 2014-12-01 to 2019-01-09. It is resampled to hourly intervals.

## Usage
1. Clone the repository.
2. Ensure all required packages are installed.
3. Place the dataset (coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv.zip) in the project directory.
4. Run ```unzip coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv.zip```
5. Run forecast_btc.py to preprocess data and train the model.
## Modules
### Data Cleaning
- Converts Unix timestamps to datetime.
- Resamples data to hourly intervals.
- Removes data before 2015-01-26 due to NaNs.
- Interpolates remaining NaNs.
- Sequence Generation
- Creates sequences of length 24 for LSTM input.
### Data Processing
- Normalizes data between 0 and 1.
- Splits data into training, validation, and test sets.
### Model Training
- Defines and trains an LSTM model.
- Visualizes true vs. predicted values.
