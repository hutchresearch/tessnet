import os
import hydra
from datavis import DataVisualizer
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table


CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           '..', 'configs'))

SCATTER = False
CFM = True


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base='1.2')
def main(config):
    args = config.datavis
    dv = DataVisualizer(args)
    #f_name = '08Mar-18_35_15_predictions.csv'
    #f_name = '09Mar-21_33_02_predictions.csv'
    #f_name = '12Mar-15_56_41_predictions.csv'
    f_name = 'eval_predictions.csv'

    subset_commands = [None, 'tmag<=8', '(tmag>=8)&(tmag<=10)', '(tmag>=10)&(tmag<=12)', '(tmag>=12)&(tmag<=14)',
                       'tmag>=14', 'sector==1', 'sector==14', 'sector==18', 'sector==21', 'sector==12']

    scatter_metrics = []
    cfm_metrics = []
    for subset_command in subset_commands:
        if SCATTER:
            _, metric = dv.generate_fig('scatter', f_name=f_name, subset_command=subset_command, save=True)
            scatter_metrics.append(metric)
        if CFM:
            _, metric = dv.generate_fig('cfm', f_name=f_name, subset_command=subset_command, cmap='Blues', save=True)
            cfm_metrics.append(metric)

    metrics = []
    indexes = []
    if SCATTER:
        metrics.append(scatter_metrics)
        indexes.append('MSE')
    if CFM:
        metrics.append(cfm_metrics)
        indexes.append('Accuracy')

    subset_commands[0] = 'all'
    df = pd.DataFrame(metrics, index=indexes, columns=subset_commands).transpose()

    print(df)

if __name__ == '__main__':
    main()