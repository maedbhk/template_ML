import os
import numpy as np

def pydraml():
    """Parameters in `base_info` and `spec_info` are required by pydraml pipeline: https://github.com/nipype/pydra-ml 
    """

    base_info = {
        "filename" : None,
        "x_indices" : None,
        "target_vars" : None,
        "permute" : [True, False],
        "group_var" : None,
        "n_splits" : 5,
        "test_size" : .2,
        "permute" : [True, False],
        "gen_feature_importance" : True,
        "gen_permutation_importance" : True,
        "permutation_importance_n_repeats" : 5,
        "permutation_importance_scoring" : "accuracy",
        "gen_shap" : False,
        "nsamples" : "auto",
        "l1_reg" : "aic",
        "plot_top_n_shap": 10,
        "metrics" : ['roc_auc_score', 'f1_score', 'precision_score', 'recall_score']
        }

    spec_info = {
        'pydraml1':
        {'clf_info': 
        [
            ["sklearn.ensemble", "AdaBoostClassifier"],
            ["sklearn.naive_bayes", "GaussianNB"],
            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}],
            ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100}],
            ["sklearn.ensemble", "ExtraTreesClassifier", {"n_estimators": 100, "class_weight": "balanced"}],
            ["sklearn.linear_model", "LogisticRegressionCV", {"solver": "liblinear", "penalty": "l1"}],
            ["sklearn.neural_network", "MLPClassifier", {"alpha": 1, "max_iter": 1000}],
            ["sklearn.svm", "SVC", {"probability": True},
            [{"kernel": ["rbf", "linear"], "C": [1, 10, 100, 1000]}]],
        ]
        },
        'pydraml2':
        {'clf_info': 
        [
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}]], # classifier has to be last list
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.linear_model", "LogisticRegressionCV", {"solver": "saga", "penalty": "l1", "max_iter": 5000}]], # classifier has to be last list
        [["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 50}]] # classifier has to be last list
        ],
        },
        }
    return base_info, spec_info


def participants():
    """Any variables set in `spec_info` (e.g., `age`, `sex`, `race`) should be present in `base_info.filename` 
    """

    base_info = {'filename': 'participants_train_test.csv',
                'participant_id': 'Identifiers' # should be string (e.g., 'Identifiers', 'pat_id') and it's assumed that the `participant_id` column is in the features and targets dataframes (code checks this)
                }
    
    spec_info = {
        'participant1':
        {
         'split': ['train'],
         'age': [int(t) for t in np.arange(5,22)],
         'Sex': ['male'],
         'race': ['Black/African American', 'White/Caucasian'],
        },
        }

    return base_info, spec_info


def targets():
    """hardcode target features
    """

    base_info = {
        'upsample': True, # upsample minority target class using SMOTE
                    }
    spec_info = {'target1':
                    {
                    'filename': 'Clinical_Diagnosis_Demographics.csv',
                    'target_column': 'Sex', # should be string (e.g., 'age', 'diagnosis')
                    'binarize': True,
                    'cols_to_keep': ['Identifiers', 'Sex'], # columns we want in the final dataframe
                    },
                }

    return base_info, spec_info


def features():
    base_info = {
                "threshold": False, # threshold dataframe based on some fixed criterion. We are using 50% for columns and 20% for rows. If threshold is False, then only NaN entries are removed (no thresholding applied)
                "clf_info": {
                    "numeric": [
                        [
                            "sklearn.impute",
                            "SimpleImputer",
                            {
                                "strategy": "mean"
                            }
                        ],
                        [
                            "sklearn.preprocessing",
                            "StandardScaler",
                            {}
                        ]
                    ],
                    "category": [
                        [
                            "sklearn.impute",
                            "SimpleImputer",
                            {
                                "strategy": "constant" # most_frequent,
                                "fill_value": "missing"
                            }
                        ],
                        [
                            "sklearn.preprocessing",
                            "OneHotEncoder",
                            {
                                "handle_unknown": "ignore",
                                "sparse": False
                            }
                        ]
                    ]
                    }
        }

    spec_info = {
            'features1':
            {
            "filename": 'Clinical_Diagnosis_Demographics.csv', 
            "cols_to_drop": [], # cols to drop from dataframe
            }, 
    }

    return base_info, spec_info




