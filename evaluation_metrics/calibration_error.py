import numpy as np
import calibration as cal
import numpy as np
import matplotlib.pyplot as plt

"""
## Reference for calibration library: https://pypi.org/project/uncertainty-calibration/

@inproceedings{kumar2019calibration,
  author = {Ananya Kumar and Percy Liang and Tengyu Ma},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  title = {Verified Uncertainty Calibration},
  year = {2019}}
"""


def calculate_calibration_error(y_true, y_pred, n_bins=10, save_curve_filepath=None):
    """Calculate Expected Calibration Error (ECE) for probabilistic predictions.
    Two methods are provided to verify the calculation:
    1. Manual calculation of ECE.
    2. Using the `calibration` library's get_ece function.

    Args:
        y_true (array-like): True binary labels (0 or 1)
        y_pred (array-like): Predicted probabilities for the positive class
        n_bins (int): Number of bins to use for calibration curve
        save_curve_filepath (str or None): If provided, saves reliability diagram to this path

    Returns:
        float: Expected Calibration Error (ECE) value
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, 0, 1)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_pred, bin_edges[1:-1], right=True)

    ece = 0.0
    N = len(y_true)

    # Collect bin stats for optional plotting
    mean_conf = np.zeros(n_bins, dtype=float)
    mean_acc = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        counts[b] = count

        if count > 0:
            acc = float(y_true[mask].mean())
            conf = float(y_pred[mask].mean())
            mean_acc[b] = acc
            mean_conf[b] = conf
            ece += abs(acc - conf) * count
        else:
            mean_acc[b] = 0.0
            mean_conf[b] = 0.0

    ece /= N

    # Verify with calibration library
    ece_lib = cal.get_ece(y_pred, y_true, num_bins=n_bins)
    assert np.isclose(ece, ece_lib), f"ECE mismatch: calculated={ece} vs library={ece_lib}"

    # reliability diagram saving
    if save_curve_filepath is not None:
        fig, ax = plt.subplots()

        nbins = len(mean_conf)
        xticks = [x + 0.5 for x in range(nbins)]

        # ax.bar(xticks, mean_conf, label='Confidence')
        ax.bar(xticks, mean_acc, label='Accuracy')
        ax.plot((0.0, nbins), (0.0, 1.0), 'k')
        ax.set_xlabel('Confidence bin')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(xticks)
        xticklabels = tuple(['{:.1f}'.format(x / float(nbins)) for x in range(nbins)])
        ax.set_xticklabels(xticklabels)

        for i in range(len(mean_acc)):
            ax.text(
                xticks[i],
                mean_acc[i],
                '{:d}'.format(int(counts[i])),
                horizontalalignment='center',
                fontsize=10
            )

        plt.xlim((0, nbins))
        fig.tight_layout()
        fig.savefig(save_curve_filepath, dpi=300)
        plt.close(fig)

    return ece