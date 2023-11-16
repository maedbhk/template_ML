import click
import warnings
warnings.filterwarnings("ignore")

def run(
    model_spec, 
    features,
    out_dir=None,
    cache_dir=None
    ):
    """ run predictive models using pydra-ml. must provide `model_spec` (model parameters outlined here) json and `features` (csv of features)

    Args:
        model_spec (str): full path to model spec file.
        features (str): fullpath to features csv (should correspond to filename coded in `model_spec`)
        out_dir (str or None): full path to model output directory. Default is home directory.
        cache_dir (str or None): fullpath to cache directory for pydra-ml intermediary outputs. Default is home directory.
    Returns: 
        saves (pickled) model to `out_dir`
    """
    # load libraries
    import os
    from src import io
    from pathlib import Path
    from pydra_ml.classifier import gen_workflow, run_workflow

    # make `cache_dir`
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + '/.cache/pydra-ml/cache-wf/'

    # make `out_dir`
    if out_dir is None:
        out_dir = os.path.expanduser('~')

    # create `cache_dir` and `out_dir` if it hasn't already been created
    io.make_dirs(cache_dir)
    io.make_dirs(out_dir)

    # load model spec json
    spec_info = io.load_json(model_spec)

    print(f'running {model_spec}...\n', flush=True)
    print("spec info", spec_info, flush=True)

    # assign fullpath to features csv
    spec_info['filename'] = features

    # change directory to output directory
    os.chdir(out_dir)
    print(f'changing directory to {out_dir}')

    # run workflow
    wf = gen_workflow(spec_info, cache_dir=cache_dir)
    run_workflow(wf, "cf", {"n_procs": 1})


if __name__ == "__main__":
    run()


    