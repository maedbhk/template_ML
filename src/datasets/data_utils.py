# useful functions for manipulating data

import pandas as pd

def remove_small_groups(dataframe, columns_to_stratify, min_group_size=2):
    """Removes the rows from the dataframe where the group size is less than min_group_size

    Args:
        dataframe (pd.DataFrame): The dataframe to remove the rows from
        columns_to_stratify (list): A list of columns to use for grouping

    Returns:
        pd.DataFrame: The dataframe with the small groups removed
    """

    # Group the dataframe by the columns_to_stratify columns
    grouped_dataframe = dataframe.groupby(columns_to_stratify)

    # Filter the grouped dataframe to only include groups with at least min_group_size rows
    filtered_dataframe = grouped_dataframe.filter(lambda x: len(x) >= min_group_size)

    # Remove the rows from the dataframe that are not in the filtered dataframe
    dataframe = dataframe[dataframe.index.isin(filtered_dataframe.index)]

    # Return the filtered dataframe
    return dataframe