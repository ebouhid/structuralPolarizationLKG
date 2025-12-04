import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_data(input_path: str, train_ratio: float = 0.7, valid_ratio: float = 0.15, test_ratio: float = 0.15, random_state: int = 42):
    """
    Split a tabular dataframe into train, validation and test sets.
    
    Args:
        input_path: Path to input file (.csv or .parquet)
        train_ratio: Proportion of data for training set
        valid_ratio: Proportion of data for validation set
        test_ratio: Proportion of data for test set
        random_state: Random seed for reproducibility
    """
    # Validate ratios
    if not abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")
    
    # Parse input path
    input_path_str = Path(input_path)
    if not input_path_str.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read data based on file extension
    extension = input_path_str.suffix.lower()
    if extension == '.csv':
        df = pd.read_csv(input_path_str)
    elif extension == '.parquet':
        df = pd.read_parquet(input_path_str)
    else:
        raise ValueError(f"Unsupported file format: {extension}. Use .csv or .parquet")
    
    # Split data
    # First split: separate test set
    train_valid_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    valid_size_adjusted = valid_ratio / (train_ratio + valid_ratio)
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_size_adjusted,
        random_state=random_state
    )
    
    # Save splits to the same directory as input
    output_dir = input_path_str.parent
    
    if extension == '.csv':
        train_df.to_csv(output_dir / f"train{extension}", index=False)
        valid_df.to_csv(output_dir / f"valid{extension}", index=False)
        test_df.to_csv(output_dir / f"test{extension}", index=False)
    elif extension == '.parquet':
        train_df.to_parquet(output_dir / f"train{extension}", index=False)
        valid_df.to_parquet(output_dir / f"valid{extension}", index=False)
        test_df.to_parquet(output_dir / f"test{extension}", index=False)
    
    print(f"Data split completed:")
    print(f"  Train: {len(train_df)} samples -> {output_dir / f'train{extension}'}")
    print(f"  Valid: {len(valid_df)} samples -> {output_dir / f'valid{extension}'}")
    print(f"  Test: {len(test_df)} samples -> {output_dir / f'test{extension}'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split tabular data into train, validation and test sets")
    parser.add_argument("input_path", type=str, help="Path to input file (.csv or .parquet)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio (default: 0.7)")
    parser.add_argument("--valid_ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    split_data(
        args.input_path,
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
        args.random_state
    )
