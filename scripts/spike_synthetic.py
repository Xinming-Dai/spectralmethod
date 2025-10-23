import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
from typing import Literal, Tuple
import scripts.tools as tools

class Template:
    def __init__(self, num_templates:int=3, num_channels:int=1):
        """
        Generate synthetic intracellular spikes (action potential)
        Basic shape: sharp depolarization (up), then slower repolarization and undershoot

        Args:
            num_templates (int, optional): number of templates. Defaults to 3.
            num_channels (int, optional): number of channels. Defaults to 1.
        """
        self.num_channels = num_channels #TODO: add more channels
        self.num_templates = num_templates
        self.template = None # shape (num_templates, snippet_length)

        self.template_center: int = int(42) # align at the depolarization peak
        self.snippet_length: int = int(121) # total length of the snippet
        self.time = np.arange(self.snippet_length) # the squence of time points

    def _gaussian(self, x:npt.NDArray|float, mu:float, sigma:float)->npt.NDArray|float:
        """generate Gaussian function

        Args:
            x (np.ndarray | float): input array
            mu (float): mean
            sigma (float): standard deviation

        Returns:
            np.ndarray | float: Gaussian function evaluated at x
        """
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def generate_template(self, seeds:int=None)->npt.NDArray:
        """generate synthetic intracellular action potential templates

        Args:
            seeds (int, optional): random seed for reproducibility. Defaults to None.
        Returns:
            np.ndarray: shape (num_templates, num_timepoints)
        """
        np.random.seed(seeds)
        amplitudes_depolarization = np.random.randint(2, 6, size=self.num_templates)
        amplitudes_repolarization = np.random.randint(2, 6, size=self.num_templates)
        
        template = np.zeros((self.num_templates, self.snippet_length))
        template += amplitudes_depolarization[:, None] * self._gaussian(self.time, 35, 3)[None, :] 
        template -= amplitudes_repolarization[:, None] * self._gaussian(self.time, 42, 5)[None, :] 
        template += self._gaussian(self.time, 50, 10)[None, :]
        self.template = template

        return template


    
    def plot(self)->None:
        if self.template is None:
            raise ValueError("Call generate_template() first.")
        
        tools.Plotter.plot_template(self.template)


class SpikeSynthetic:
    def __init__(self, 
                 noise_std: float = 0.5, 
                 num_templates: int = 1, 
                 template_obj: Template = None, 
                 duration: int = 60, 
                 random_state: int = 42):
        """
        Synthesize extracellular spikes from intracellular action potential template
        
        Args:
            noise_std (float): standard deviation of background noise
            num_templates (int): number of templates
            template_obj (Template): Template object containing intracellular action potential templates
            duration (int): 60 seconds by default
            random_state (int, optional): random seed for reproducibility. Defaults to 42.
        """
        self.duration = duration
        self.num_channels: int = int(1)
        self.sampling_rate: int = int(30000)
        self.num_templates = num_templates
        self.rng = np.random.default_rng(random_state)

        if template_obj is None or not isinstance(template_obj, Template):
            self.template_obj = Template(self.num_templates)
            self.template_obj.generate_template(seeds=random_state)
        else:
            self.template_obj = template_obj
        if self.template_obj.template.shape[0] < self.num_templates:
            warnings.warn("Not enough templates in the Template object. Use all templates.", UserWarning)
            self.num_templates = self.template_obj.template.shape[0]
        
        self.noise_std: float = noise_std
        self.noise: npt.NDArray[np.float16] = None        # background noise (recording_len,)
        self.raw_waveforms: npt.NDArray[np.float16] = None         # noisy data (num_channels, recording_len). Assume one channel for now. (1, recording_len)
        self.spike_clean: npt.NDArray[np.float16] = None  # clean data (without noise). shape = (num_templates, recording_len). The last spike may be cut off.
        self.spike_times: npt.NDArray[np.int_] = None  # when the spikes occur (in samples). shape = [num_templates, max_spikes_per_template]. If the spike times = -1, then it is invalid.
        self.T: int = int(round(self.duration * self.sampling_rate))  # recording_len
        self.len_template: int = self.template_obj.snippet_length     # length of each template

    def _generate_spike_times(self, T_with_extra_sample:int)->npt.NDArray[np.int_]:
        """generate spike times for each template
        Args:
            T_with_extra_sample (int): total time with extra samples to avoid boundary effects
        Returns:
            npt.NDArray[np.int_]: shape = (num_templates, max_spikes_per_template)
            spike times for each template in samples
        
        Example:
            get_spike_times_for_template_i(i) to get valid spike times for template i
        """
        isi_low: int = int(121)  # inter-spike interval
        isi_high: int = int(500)
        mean_isi = (isi_low + isi_high) / 2
        expected_spikes = int(self.duration * 1000 / mean_isi)

        isis = self.rng.integers(low=isi_low, high=isi_high, 
                                 size=(self.num_templates, expected_spikes))
        self.spike_times = np.cumsum(isis, axis=1)          # self.spike_times[i] is a list of spike times for template i
        mask = self.spike_times < T_with_extra_sample - 200 # remove last 200 samples to avoid boundary effects. These time stamps will be with in [0, self.T].
        INVALID = -1
        self.spike_times = np.where(mask, self.spike_times, INVALID)
        return self.spike_times

    def get_spike_times_for_template_i(self, i: int)->npt.NDArray[np.int_]:
        """get valid spike times for template i
        Args:
            i (int): template index
        Returns:
            npt.NDArray[np.int_]: valid spike times for template i. shape (num_spikes_i, )
        """
        if self.spike_times is None:
            raise ValueError("Please run _generate_spike_times() first.")
        INVALID = -1
        valid = self.spike_times[i] != INVALID
        return self.spike_times[i, valid]

    def generate_spike(self)->npt.NDArray[np.float16]:
        """generate synthetic extracellular spikes from intracellular action potential template

        returns:
            np.ndarray[np.float16]: shape (num_channels, recording_len)
                Noisy extracellular spike raw_waveforms
        """
        extra_sample:int = 200 # add extra sample in the end to avoid boundary effects. It will be deleted.
        T_with_extra_sample = self.T + extra_sample

        # TODO: change the noise to be stationary noise
        self.noise = self.rng.normal(0, self.noise_std, size = T_with_extra_sample)
        self.spike_clean = np.zeros(shape = (self.num_templates, T_with_extra_sample), dtype=np.float16)
        self.spike_times = self._generate_spike_times(T_with_extra_sample)

        for i in range(self.num_templates):
            spike_times_for_template_i = self.get_spike_times_for_template_i(i)
            starts = spike_times_for_template_i - self.template_obj.template_center
            idx = starts[:, None] + np.arange(self.len_template)[None, :] # shape (num_spikes, len_template)

            np.add.at(self.spike_clean[i], idx,
                      self.template_obj.template[i][None, :])
        
        self.spike_clean = self.spike_clean[:, 0:self.T] # remove last 200 samples to avoid boundary effects
        self.noise = self.noise[:self.T]
        self.raw_waveforms = self.spike_clean.sum(axis=0) + self.noise
        self.raw_waveforms = self.raw_waveforms.reshape(1, -1)  # shape (num_channels, recording_len). Assume one channel for now.
        return self.raw_waveforms

    def _get_valid_idx(self, template_idx: int)->npt.NDArray[np.int_]:
        """Get valid index matrix to extract snippets for a given template index.

        Args:
            template_idx (int): template index 

        Returns:
            npt.NDArray[np.int_]: index matrix to extract snippets.
                Shape = (num_valid_spikes, len_template).
                Each row corresponds to one snippet.
                If no valid spikes, return None.
        """
        spike_times_i: npt.NDArray[np.int_] = self.get_spike_times_for_template_i(template_idx)
        
        starts = spike_times_i - self.template_obj.template_center
        ends = starts + self.len_template
        valid_mask = (starts >= 0) & (ends <= self.T) # skip the spikes that are not completely captured in the data
        starts = starts[valid_mask]

        if starts.size == 0:
            return None

        # Build an index matrix to extract snippets
        idx = starts[:, None] + np.arange(self.len_template)[None, :]
        return idx

    def snippets(self, choice: Literal["collision_cleaned", "raw_waveforms"])->Tuple[npt.NDArray[np.float16], npt.NDArray[np.int_]]:
        """
        Extract spike snippets from spike_clean + noise or from raw waveforms using valid spike times.

        Args:
            choice (str): "collision_cleaned" or "raw_waveforms"
                "collision_cleaned": use spike_clean + noise
                "raw_waveforms": use raw_waveforms (collisions + noise)

        Returns:
        X : npt.NDArray[np.float16]
            Extracted snippets stacked along axis 0.
            Shape = (total_spikes, num_channels * snippet_length). Assume num_channels=1 for now.
        labels : npt.NDArray[np.int_]
            Template index for each snippet (0, 1, 2, ...).
            Shape = (total_spikes, )
        """
        if choice == "collision_cleaned":
            if self.spike_clean is None or self.noise is None:
                raise ValueError("Please run generate_spike() before collecting snippets.")
        if choice == "raw_waveforms":
            if self.raw_waveforms is None:
                raise ValueError("Please run generate_spike() before collecting snippets.")

        snippets: list[npt.NDArray[np.float16]] = []
        labels: list[int] = []

        for i in range(self.num_templates):
            if choice == "collision_cleaned":
                whole_data: npt.NDArray[np.float16] = self.spike_clean[i] + self.noise
            elif choice == "raw_waveforms":
                whole_data: npt.NDArray[np.float16] = self.raw_waveforms[0] # assume one channel for now

            idx = self._get_valid_idx(i)
            if idx is None:
                continue

            # Collect all snippets at once. Each row in idx selects one snippet from cleaned
            snippets_i = whole_data[idx].astype(np.float16)
            snippets.append(snippets_i)
            labels.append(np.full(idx.shape[0], i, dtype=np.int_))

        X = np.concatenate(snippets, axis=0)
        labels = np.concatenate(labels, axis=0)
        return X, labels
    
    def weights(self, labels:npt.NDArray[np.int_])->npt.NDArray[np.float16]:
        """Compute initial weights for each template based on labels.

        Returns:
            npt.NDArray[np.float_]: shape (num_templates,)
                Initial weights for each template
        """
        weights = np.bincount(labels) / labels.shape[0]
        return weights
    
    def _plot(self, title:str, 
              attribute:Literal["spike_clean", "raw_waveforms", "noise", "each_spike", "collision_cleaned"]="spike_clean", 
              start:int|None=None, 
              end:int|None=None,
              **kwargs) -> None:
        """plot spike

        Args:
            title (str): title of the plot
            attribute (str, optional): which attribute to plot. Defaults to "spike_clean". Options: "spike_clean", "raw_waveforms", "noise", "each_spike", "collision_cleaned".
            start (int, optional): start time in ms. Defaults to None, which means 0.
            end (int, optional): end time in ms. Defaults to None, which means full duration.
            kwargs: additional arguments for "collision_cleaned", e.g., template_idx
        """
        if not hasattr(self, 'spike_clean'):
            raise ValueError("Please run generate_spike() first to create spike data.")
        
        if start is None: start = 0
        if end is None: end = self.T

        t = np.arange(start, end)
        plt.figure(figsize=(10, 4))
        plt.xlim(start, end)

        if attribute == "each_spike":
            for i in range(self.num_templates):
                plt.plot(t, self.spike_clean[i, start:end], lw=1, label=f'Template {i+1}')
            plt.legend(loc="upper right")
        else:
            if attribute == "raw_waveforms":
                y = self.raw_waveforms.flatten()
            elif attribute == "noise":
                y = self.noise
            elif attribute == "collision_cleaned":
                template_idx = kwargs.get("template_idx")
                if template_idx is None:
                    raise ValueError("Please provide 'template_idx' argument in kwargs for 'collision_cleaned' plot.")
                if not (0 <= template_idx < self.num_templates):
                    raise ValueError(f"template_idx should be in [0, {self.num_templates - 1}]")
                y = self.spike_clean[template_idx] + self.noise
            else:
                y = self.spike_clean.sum(axis=0)
            plt.plot(t, y[start:end], lw=1)
            
        plt.title(title)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (a.u.)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_spike_clean(self, start:int|None=None, end:int|None=None)->None:
        self._plot("Raw Waveforms - Background Noise (30 kHz sampling)", attribute="spike_clean", start=start, end=end)

    def plot_raw_waveforms(self, start:int|None=None, end:int|None=None)->None:
        self._plot("Raw Waveforms (30 kHz sampling)", attribute="raw_waveforms", start=start, end=end)

    def plot_noise(self, start:int|None=None, end:int|None=None)->None:
        self._plot("Background Noise (30 kHz sampling)", attribute="noise", start=start, end=end)

    def plot_each_template(self, start:int|None=None, end:int|None=None)->None:
        self._plot("Each Template Spike Train (30 kHz sampling)", attribute="each_spike", start=start, end=end)

    def plot_collision_cleaned_snippets(self, start: int | None = None, end: int | None = None, template_idx: int | None = 0) -> None:
        self._plot("Collision Cleaned Snippets", attribute="collision_cleaned", start=start, end=end, template_idx=template_idx)

    def plot_snippet_pca(self, choice: Literal["collision_cleaned", "raw_waveforms"] = "collision_cleaned") -> None:
        """
        Compute and plot the first two principal components of spike snippets.

        Args:
            choice (str): "collision_cleaned" or "raw_waveforms".
                Determines which snippets to use for PCA.
        """
        X, labels = self.snippets(choice=choice)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.title(f"First Two Principal Components ({choice})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(alpha=0.3)
        plt.legend(*scatter.legend_elements(), title="Template")
        plt.tight_layout()
        plt.show()