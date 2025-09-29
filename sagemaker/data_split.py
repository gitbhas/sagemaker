import pandas as pd
import json
from datetime import datetime, timedelta

def read_json_from_local(file_path, chunk_size=10000):
    columns_of_interest = ['fil_creatn_dt', 'fil_creatn_time', 'fil_id', 'in_tot_rec_cnt']

    chunks = []
    with open(file_path, 'r') as f:
        chunk = []
        for i, line in enumerate(f):
            chunk.append(json.loads(line))
            if (i + 1) % chunk_size == 0:
                df = pd.DataFrame(chunk)[columns_of_interest]
                chunks.append(df)
                chunk = []

        # Add any remaining data
        if chunk:
            df = pd.DataFrame(chunk)[columns_of_interest]
            chunks.append(df)

    return pd.concat(chunks, ignore_index=True)

def prepare_data(df):
    print("Processing dates and times...")
    # Convert the time format from HH.MM.SS to HH:MM:SS
    df['fil_creatn_time'] = df['fil_creatn_time'].str.replace('.', ':')
    df['datetime'] = pd.to_datetime(df['fil_creatn_dt'] + ' ' + df['fil_creatn_time'])

    print("Filtering data for the last 5 years...")
    current_year = datetime.now().year
    df = df[df['datetime'].dt.year >= (current_year - 5)]

    print("Adding AM/PM cycle...")
    df['cycle'] = df['datetime'].dt.strftime('%p')

    print("Grouping data by date and cycle...")
    daily_data = df.groupby([df['datetime'].dt.date, 'cycle']).agg({
        'fil_id': 'count',
        'in_tot_rec_cnt': 'sum'
    }).reset_index()

    print("Renaming columns...")
    daily_data.columns = ['date', 'cycle', 'file_count', 'total_records']

    print("Adding features...")
    daily_data['date'] = pd.to_datetime(daily_data['date'])  # Ensure date is datetime
    daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
    daily_data['month'] = daily_data['date'].dt.month
    daily_data['is_weekday'] = (daily_data['day_of_week'] < 5).astype(int)

    print("Sorting data...")
    daily_data = daily_data.sort_values(['date', 'cycle'])

    print("Data preparation complete.")
    print(f"Columns in the prepared data: {daily_data.columns.tolist()}")
    print(f"Shape of the prepared data: {daily_data.shape}")
    return daily_data

def split_data(prepared_data, test_months=6):
    split_date = prepared_data['date'].max() - pd.DateOffset(months=test_months)
    train_data = prepared_data[prepared_data['date'] < split_date]
    test_data = prepared_data[prepared_data['date'] >= split_date]
    return train_data, test_data

if __name__ == "__main__":
    file_path = 'file_audit.json'  # Replace with the actual path to your local file

    try:
        print("Starting data preparation process...")

        data = read_json_from_local(file_path)
        print(f"Successfully read {len(data)} records from the local file.")

        prepared_data = prepare_data(data)
        print(f"Prepared data shape: {prepared_data.shape}")

        print("First few rows of prepared data:")
        print(prepared_data.head())

        print("Data preparation completed successfully.")

        # Split the data for training and testing
        train_data, test_data = split_data(prepared_data)

        print(f"Training data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise