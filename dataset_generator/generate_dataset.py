from matplotlib.pylab import norm
import numpy as np
import pandas as pd            

class BernoulliPanel:
    """
    Generate and hold a panel-style dataset of Bernoulli draws per sample.

    Each sample j has a Bernoulli parameter p_j. For that sample, we draw
    N_radiologists independent Bernoulli outcomes:
        y1_j, y2_j, ..., yN_j ~ Bernoulli(p_j)

    Attributes
    ----------
    n_radiologists : int
        Number of draws (columns y1..yN) per sample.
    n_samples : int
        Number of samples (rows) to generate.
    p_list : list[float]
        Per-sample Bernoulli parameters in [0, 1], length == n_samples.
    random_seed : int or None (default 42)
        RNG seed for reproducibility.
    df : pandas.DataFrame or None
        Holds the generated data with columns: 'p', 'y1', ..., 'yN'.

    Steps to use
    ----------------
    1. Create an instance of BernoulliPanel with desired n_radiologists and
       n_samples.
     2. Call generate_bimodal_power_plist(...) to generate p_list.
     3. Call generate_dataset() to create the dataframe and store it in `self.df`.
         Each call to generate_dataset() overwrites the previous dataframe.

    Examples
    --------
    Example 1: Generate dataset with bimodal p_list
    >>> panel = BernoulliPanel(n_radiologists=10, n_samples=20)
    >>> panel.generate_bimodal_power_plist(a=2.0, clip_eps=0.01, shuffle=True)
    >>> df = panel.generate_dataset()
    """

    def __init__(self, n_radiologists, n_samples, random_seed=42):
        """
        Initialize the generator; no data is created until `generate()` is called.

        Parameters
        ----------
        n_radiologists : int
            Number of Bernoulli draws per sample.
        n_samples : int
            Number of samples to generate.
        random_seed : int or None
            Seed for numpy's default_rng.
        """
        self.n_radiologists = int(n_radiologists)
        self.n_samples = int(n_samples)
        self.random_seed = random_seed
        self.p_list = None
        self.df = None
    

    def generate_bimodal_power_plist(
        self,
        a: float,
        n0: int | None = None,
        clip_eps: float | None = None,
        shuffle: bool = True,
        ):
        """
        Bimodal symmetric distribution:
        - n0 samples in (0, 0.5):  p = 0.5 * u^a
        - n1 samples in (0.5, 1):  p = 1 - 0.5 * u^a

        a controls subjectivity:
        - large a  -> mass near 0 and 1 (sharp peaks)
        - small a  -> mass shifts toward 0.5 (center fills)

        Fresh u draws per call (no linkage across worlds unless you reuse seed).
        """
        a = float(a)
        if a <= 0:
            raise ValueError("a must be > 0")

        rng = np.random.default_rng(self.random_seed)
        n = int(self.n_samples)

        n0 = (n // 2) if n0 is None else int(n0)
        if not (0 <= n0 <= n):
            raise ValueError("n0 must be in [0, n_samples]")
        n1 = n - n0

        u_neg = rng.random(n0)
        u_pos = rng.random(n1)

        p_neg = 0.5 * (u_neg ** a)
        p_pos = 1.0 - 0.5 * (u_pos ** a)
        p = np.concatenate([p_neg, p_pos]).astype(float)

        if clip_eps is not None:
            clip_eps = float(clip_eps)
            if not (0 < clip_eps < 0.5):
                raise ValueError("clip_eps must be in (0, 0.5)")
            p[:n0] = np.clip(p[:n0], clip_eps, 0.5 - clip_eps)
            p[n0:] = np.clip(p[n0:], 0.5 + clip_eps, 1.0 - clip_eps)

        if shuffle:
            rng.shuffle(p)

        self.p_list = p.tolist()
        return self.p_list

    def generate_dataset(self):
        """
        Create the dataframe and store it on `self.df`.

        Returns
        -------
        pandas.DataFrame
            A dataframe with columns:
              - 'p' : per-sample Bernoulli parameter
              - 'y1'..'yN' : Bernoulli outcomes for each radiologist
        """
        if self.p_list is None:
            msg = (
                "p_list is not set. Call generate_homogeneous_plist() or "
                "generate_heterogeneous_plist() or generate_custom_plist() first."
            )
            raise ValueError(msg)

        p_arr = np.asarray(self.p_list, dtype=float)
        if len(p_arr) != self.n_samples:
            msg = (
                "Each sample must have a p value. "
                f"Got len(p_list)={len(p_arr)}, n_samples={self.n_samples}"
            )
            raise ValueError(msg)
        if np.any((p_arr < 0) | (p_arr > 1)):
            raise ValueError("All p values must be in [0, 1]")

        rng = np.random.default_rng(self.random_seed)
        ys = rng.binomial(
            n=1,
            p=p_arr[:, None],
            size=(self.n_samples, self.n_radiologists),
        )

        df = pd.DataFrame({'p': p_arr})
        y_cols = {f"y{i+1}": ys[:, i] for i in range(self.n_radiologists)}
        df = pd.DataFrame({"p": p_arr, **y_cols})

        self.df = df
        return self.df
