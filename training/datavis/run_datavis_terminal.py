import os
import hydra
from datavis import DataVisualizer
import argparse

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           '..', 'configs'))


@hydra.main(config_path=CONFIG_PATH, config_name="config")
def main(config):
    args = config.datavis
    dv = DataVisualizer(args)

    f_name = args.f_name
    if args.fig == 'scatter':
        dv.generate_fig('scatter', f_name=f_name, show=args.show, save=args.save)
    elif args.fig == 'cfg':
        dv.generate_fig('cfm', f_name=f_name, show=args.show, save=args.save)


if __name__ == '__main__':
    main()