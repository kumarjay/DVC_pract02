import argparse
import joblib
import json
import os
import pandas as pd
from typing import Text, Dict, List
import yaml

# from src.data.dataset import get_target_names

print(os.getcwd())
print(os.listdir())
# export PYTHONPATH="$PYTHONPATH:/mnt"
import sys
sys.path.insert(0, '/home/jay/PycharmProjects/DVC_pract01')

from src.data.dataset import get_target_names
from src.evaluate.evaluate import evaluate
from src.report.visualize import plot_confusion_matrix


def evaluate_model(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))
    target_column = config['featurize']['target_column']
    model_name = config['base']['model']['model_name']
    model_folder = config['base']['model']['models_folder']
    reports_folder = config['base']['reports']['reports_folder']

    test_df = pd.read_csv(config['data_split']['test_path'])
    model = joblib.load(os.path.join(model_folder, model_name))
    report = evaluate(df=test_df,
                      target_column=target_column,
                      clf=model)
    classes = get_target_names()
    print(report['f1'])
    metrics_path = os.path.join(reports_folder, config['evaluate']['metrics_file'])
    json.dump(
        obj={'f1_score': report['f1']},
        fp=open(metrics_path, 'w')
    )
    print(f'f1 metrics file saved to : {metrics_path}')

    plt = plot_confusion_matrix(cm=report['cm'],
                                target_names=get_target_names(),
                                normalize=False,
                                config= config)
    # confusion_matrix_png_path = os.path.join(reports_folder, config['evaluate']['confusion_matrix_png'])
    # print(confusion_matrix_png_path, plt.shape)
    # plt.savefig(confusion_matrix_png_path)


    classes_path = os.path.join(reports_folder, config['evaluate']['classes_path'])
    mapping = {
        0: classes[0],
        1: classes[1],
        2: classes[2]
    }
    df = (pd.DataFrame({'actual': report['actual'],
                        'predicted': report['predicted']})
          .assign(actual=lambda x: x.actual.map(mapping))
          .assign(predicted=lambda x: x.predicted.map(mapping)))
    df.to_csv(classes_path, index=False)
    print(f'Classes actual/predicted saved to : {classes_path}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    evaluate_model(config_path=args.config)
