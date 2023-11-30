
def run(
    fpath, 
    outpath,
    columns_to_stratify=[], 
    columns_to_keep=[], 
    test_size=0.2, 
    random_state=42,
    ):
    """Splits a Pandas DataFrame into train and test sets using stratified sampling, handling the case where some of the groups in the columns_to_stratify list have only 1 value.

    Args:
        fpath (str): Fullpath to dataframe to split.
        outpath (str): Fullpath to file where the train and test sets will be saved.
        columns_to_stratify (list): A list of columns to use for stratified sampling. Defaults to empty list.
        columns_to_keep (list): A list of columns to keep in the train and test sets. Defaults to empty list.
        test_size (float): The proportion of the data to be included in the test set. Defaults to 0.2.
        random_state (int): The random seed to use for splitting the data. Defaults to 42.

    Returns:
        train_df (pandas.DataFrame): The training set.
        test_df (pandas.DataFrame): The test set.
    """
    import re
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.datasets.data_utils import remove_small_groups

    # read in dataframe from path
    df = pd.read_csv(fpath)
    
    # remove small groups from dataframe (otherwise won't be able to stratify)
    df = remove_small_groups(dataframe=df, columns_to_stratify=columns_to_stratify)

    stratify = None
    if len(columns_to_stratify) > 0:
        stratify = df[columns_to_stratify]

    # Split the DataFrame into train and test sets using stratified sampling.
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)

    # Assign split and concat dataframes
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    df_out = pd.concat([train_df, test_df])

    # always save out column: 'split'
    columns_to_keep.append('split')

    # Index into dataframe using `columns_to_keep`
    if len(columns_to_keep) > 1:
        df_out = df_out[columns_to_keep]

    # save dataframe to `out_dir`
    df_out.reset_index(drop=True).to_csv(outpath, index=False)
    print(f'train/test splits saved to {outpath}', flush=True)

    