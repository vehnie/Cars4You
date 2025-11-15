import pandas as pd


def drop_redundant_columns(df, columns=['hasDamage','previousOwners', 'paintQuality%']):
    """Drop columns that are redundant or not useful for analysis."""
    df = df.drop(columns=columns, errors='ignore')
    return df

def rename_columns(df, rename_dict={'carID': 'car_id','Brand': 'brand', 'fuelType': 'fuel_type', 'engineSize': 'engine_size'}):
    """Rename columns for consistency and ease of use."""
    df = df.rename(columns=rename_dict)
    return df


def fix_negative_values(df, columns=['mileage', 'tax', 'engine_size']):
    """Fix negative values in specified columns by taking their absolute values."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].abs()
    return df

def fix_extreme_values(df):
    df = df[df['year'] > 1970]

    df['mileage'] = df['mileage'].clip(lower=0, upper=175000)

    return df