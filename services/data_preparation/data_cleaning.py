import pandas as pd

def clean_dataset(dataset_path):
    # Read the dataset, handle encoding issues
    df = pd.read_csv(dataset_path, encoding='unicode_escape')
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace('[^a-zA-Z0-9]', '_')
    
    # Clean Quantity column
    df['quantity'] = df['quantity'].str.replace('[^0-9]', '').astype(int)
    
    # Clean Unit Price column
    df['unit_price'] = df['unit_price'].str.replace('[^0-9.]', '').astype(float)
    
    # Clean InvoiceDate column
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    
    # Remove rows with missing values
    df.dropna(inplace=True)
    
    # Clean CustomerID
    df['customer_id'] = df['customer_id'].astype(str).str.replace('[^0-9]', '')
    
    # Clean Country column
    df['country'] = df['country'].str.replace('[^a-zA-Z ]', '')
    
    # Save the cleaned dataset
    cleaned_dataset_path = 'cleaned_dataset.csv'
    df.to_csv(cleaned_dataset_path, index=False)
    
    return cleaned_dataset_path
