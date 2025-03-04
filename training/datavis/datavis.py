import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, accuracy_score
from scipy.stats import gaussian_kde
import os
import scipy
from KDEpy import FFTKDE


class DataVisualizer:
    def __init__(self, args):
        # save defaults from hydra args
        self.input_data_dir = args.input_data_dir
        self.fig_save_dir = args.fig_save_dir
        self.scale = args.scale
        self.color_func = args.color_func

        # names of dataframe columns
        self.reg_x_label = args.reg_x_label
        self.reg_y_label = args.reg_y_label
        self.class_x_label = args.class_x_label
        self.class_y_label = args.class_y_label
        self.subset_command = args.subset_command
        self.sector_label = args.sector_label

        self.cmap = plt.cm.get_cmap(args.cmap)
        self.tmag_label = args.tmag_label
        self.labels = [self.reg_x_label, self.reg_y_label, self.class_x_label, self.class_y_label,
                       self.sector_label, self.tmag_label]
        self.pt_size = args.pt_size
        self.reg_x = None
        self.reg_y = None
        self.sectors = None
        self.tmags = None
        self.class_x = None
        self.class_y = None
        self.dataset_len = None

    def generate_fig(self, fig_type: str, df=None, f_name=None,
                     subset_command=-1, cmap=None,
                     save=False, show=False, **kwargs):
        if df is None and f_name is None:
            raise ValueError("One of df or f_name must be set.")

        if f_name is not None:
            fpath = self.input_data_dir + '/' + f_name
            df = pd.read_csv(fpath)

        fig, ax = plt.subplots()

        if cmap is not None:
            self.cmap = cmap
        # allow subset_command to be None
        if subset_command == -1:
            subset_command = self.subset_command

        if subset_command is not None:
            eval_command = subset_command
            for label in self.labels:
                eval_command = eval_command.replace(label, "df['{}']".format(label))

            # unsafe, general except statement for eval, but this is so much easier than validating input
            try:
                cond = eval(eval_command)
            except:
                raise ValueError("Error in usage of subset command")

            df = df.where(cond).dropna()

        # load columns of dataframe
        self.reg_x = df[self.reg_x_label].to_numpy()
        self.reg_y = df[self.reg_y_label].to_numpy()
        self.class_x = df[self.class_x_label].to_numpy()
        self.class_y = df[self.class_y_label].to_numpy()
        self.sectors = df[self.sector_label].to_numpy()
        self.tmags = df[self.tmag_label].to_numpy()

        if fig_type == 'scatter':
            metric = mean_squared_error(self.reg_y, self.reg_x)
            metric_label = 'MSE: {:.3f}'.format(metric)
        elif fig_type == 'cfm':
            metric = accuracy_score(self.class_x, self.class_y)
            metric_label = 'Accuracy: {:.3f}'.format(metric)
        else:
            raise ValueError

        if subset_command is not None:
            ax.set_title('{} {}'.format(subset_command, metric_label))
        else:
            ax.set_title('All: {}'.format(metric))

        self.dataset_len = len(self.reg_x)
        if fig_type == 'scatter':
            self._generate_scatter(fig, ax, **kwargs)
        elif fig_type == 'cfm':
            self._generate_cfm(fig, ax, **kwargs)

        if save:
            i = 0
            fname = None
            while fname is None or os.path.isfile(fname):
                fname = os.path.join(self.fig_save_dir, '{}-{}.png'.format(fig_type, i))
                i += 1
            fig.savefig(fname)
        if show:
            fig.show()
        return fig, metric

    def _generate_cfm(self, fig, ax):
        cfm = confusion_matrix(self.class_y, self.class_x)
        cfm_d = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=[0, 1, 2])
        cfm_d.plot(values_format='', cmap=self.cmap, ax=ax)

    def _generate_scatter(self, fig, ax,
                          scale=None,
                          color_func=None,
                          pt_size=None):

        # set defaults if no override given
        if scale is None:
            scale = self.scale
        if color_func is None:
            color_func = self.color_func
        if pt_size is None:
            pt_size = self.pt_size

        ax.set_xscale(scale)
        ax.set_yscale(scale)

        ax.set_xlabel(self.reg_y_label)
        ax.set_ylabel(self.reg_x_label)

        split_c_funcs = {
            'class_split_true': {
                'split_fn': lambda: self.class_y,
                'label': 'true_class'
            },
            'class_split_pred': {
                'split_fn': lambda: self.class_x,
                'label': 'pred_class'
            },
            'class_split_error': {
                'split_fn': lambda: [0 if self.class_x[i] == self.class_y[i] else 1 for i in range(self.dataset_len)],
                'label': 'class_error'
            },
            'sector_split': {
                'split_fn': lambda: self.sectors,
                'label': 'sector'
            },
            'tmag_split': {
                'split_fn': lambda: self.tmags,
                'label': 'tmag'
            }
        }

        if color_func == 'kde':
            reg_xy = np.vstack([self.reg_x, self.reg_y])
            z = gaussian_kde(reg_xy)(reg_xy)
            ax.scatter(self.reg_y, self.reg_x, c=z, s=pt_size)

        elif color_func == 'fftkde':
            data = np.transpose(np.vstack((self.reg_y, self.reg_x)))
            labels_linear_grid, grid_probs = FFTKDE(kernel="gaussian", bw=1).fit(data).evaluate(2**8)
            interp = scipy.interpolate.LinearNDInterpolator(labels_linear_grid, grid_probs)
            labels_probs = interp(self.reg_y, self.reg_x)

            ax.scatter(self.reg_y, self.reg_x, c=labels_probs, s=pt_size)

        elif color_func in split_c_funcs:
            fn_info = split_c_funcs[color_func]
            split = fn_info['split_fn']()
            scat = ax.scatter(self.reg_y, self.reg_x, s=pt_size, c=split, cmap=self.cmap)
            cb = fig.colorbar(scat, spacing='uniform')
            cb.set_label(fn_info['label'])

        else:
            ax.scatter(self.reg_y, self.reg_x, s=pt_size)

