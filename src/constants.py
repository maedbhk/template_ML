from pathlib import Path, PosixPath
import os

class Defaults:

    ## set directories for feature and model specs
    BASE_DIR = Path(__file__).absolute().parent.parent 
    FIG_DIR = BASE_DIR / "reports" / "figures"
    FEATURE_DIR = BASE_DIR / 'features'
    MODEL_SPEC_DIR = BASE_DIR / 'model_specs'
    BASH_SCRIPTS = BASE_DIR / 'bash_scripts'

    DATA_DIR = BASE_DIR / 'data'
    RAW_DIR = DATA_DIR / "raw"
    INTERIM_DIR = DATA_DIR / "interim"
    PROCESSED_DIR = DATA_DIR / "processed"

    dirs = [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR]
    for dirn in dirs:
        if not os.path.isdir(dirn):
            try:
                os.makedirs(dirn)
                print(f'make new dir: {dirn}')
            except:
                print(f'could not make {dirn}')
