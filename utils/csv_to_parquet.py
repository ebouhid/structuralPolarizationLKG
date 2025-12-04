import pandas as pd
import argparse

def csv_to_parquet(csv_file_path, parquet_file_path=None):
    """
    Convert a CSV file to Parquet format.
    
    Args:
        csv_file_path: Path to the input CSV file
        parquet_file_path: Path to the output Parquet file (optional)
    """
    if parquet_file_path is None:
        parquet_file_path = csv_file_path.replace('.csv', '.parquet')
    
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    
    # Write to Parquet format
    df.to_parquet(parquet_file_path, index=False, engine='pyarrow')
    
    print(f"Converted {csv_file_path} to {parquet_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a CSV file to Parquet format.')
    parser.add_argument('csv_file', help='Path to the input CSV file')
    parser.add_argument('parquet_file', nargs='?', default=None, 
                        help='Path to the output Parquet file (optional)')
    
    args = parser.parse_args()
    
    csv_to_parquet(args.csv_file, args.parquet_file)
