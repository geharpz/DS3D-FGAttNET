# src/utils/config_loader.py

import yaml
from types import SimpleNamespace

class ConfigLoader:
    """
    Loads a YAML configuration file and allows attribute-style access.

    Attributes
    ----------
    config_path : str
        Path to the YAML configuration file.

    Methods
    -------
    load_config()
        Loads and parses the YAML configuration.
    _dict_to_namespace(d)
        Recursively converts a dictionary into a SimpleNamespace.
    """

    def __init__(self, config_path: str):
        """
        Parameters
        ----------
        config_path : str
            Path to the YAML file containing configurations.
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """
        Loads the YAML file and converts it into a nested namespace.

        Returns
        -------
        SimpleNamespace
            Configuration object with attribute-style access.
        """
        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return self._dict_to_namespace(config_dict)

    def _dict_to_namespace(self, d):
        """
        Recursively converts a dictionary to a SimpleNamespace.

        Parameters
        ----------
        d : dict
            Dictionary to convert.

        Returns
        -------
        SimpleNamespace
            The converted object.
        """
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self._dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self._dict_to_namespace(i) for i in d]
        else:
            return d

    def get(self):
        """
        Returns the configuration object.

        Returns
        -------
        SimpleNamespace
            Configuration object.
        """
        return self.config
