import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
import utils


class ConfigManager:
    """
    Configuration manager that loads settings based on the cluster environment.
    Determines the environment by checking for 'mpcdf' or 'mpi' variables.
    """

    def __init__(
        self,
        config_dir: str = "env_config",
        mpi_config_path: str = "mpi_config.json",
        mpcdf_config_path: str = "mpcdf_config.json",
        verbose: bool = True,
    ):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory where configuration files are stored (relative to script)
            mpi_config_path: Path to MPI SLURM configuration file
            mpcdf_config_path: Path to MPCDF SLURM configuration file
            verbose: Whether to print detailed logs (default: True)
        """
        self.verbose = verbose
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            if utils.is_main_process():
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            else:
                self.logger.addHandler(logging.NullHandler())
                
        # Set log level based on verbose flag
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        self.logger.debug("Initializing ConfigManager")

        self.config_dir = Path(config_dir)
        self.logger.debug(f"Configuration directory set to: {self.config_dir}")

        self.mpi_config_path = mpi_config_path
        self.mpcdf_config_path = mpcdf_config_path

        self._get_config()
        
    def detect_environment(self) -> str:
        """
        Detect the current cluster environment.
        
        Returns:
            str: 'mpcdf' or 'mpi' or 'unknown'
        """
        self.logger.debug("Detecting environment...")

        if os.environ.get("mpcdf") is not None:
            self.logger.debug("Detected MPCDF environment")
            return "mpcdf"
        elif os.environ.get("mpi") is not None:
            self.logger.debug("Detected MPI environment")
            return "mpi"
        else:
            self.logger.error("Environment not detected! Missing environment variables.")
            raise ValueError(
                "Environment not detected. Please set either 'mpcdf' or 'mpi' environment variable."
            )

    def _get_config(self) -> Dict[str, Any]:
        """
        Get configuration based on detected environment.

        Returns:
            Dict[str, Any]: Configuration dictionary

        Raises:
            RuntimeError: If environment cannot be detected and no default is specified
        """
        self.logger.debug("Getting configuration based on environment")

        try:
            env = self.detect_environment()

            # Load appropriate config file
            if env == "mpcdf":
                config_path = self.config_dir / self.mpcdf_config_path
            elif env == "mpi":
                config_path = self.config_dir / self.mpi_config_path
            else:
                self.logger.error(f"Unknown environment: {env}")
                raise RuntimeError(
                    "Unknown environment. Please set either 'mpcdf' or 'mpi' "
                    "environment variable, or specify a default configuration."
                )

            self.logger.debug(f"Loading configuration from: {config_path}")
            self.logger.info(f"Loading configuration from: {config_path}")

            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                    self.logger.debug("Configuration loaded successfully")
                    self.logger.info("Configuration loaded successfully.")
                    self.logger.debug(f"Config contents: {self.config}")
                    return self.config
            except FileNotFoundError:
                self.logger.error(f"Configuration file not found: {config_path}")
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON in configuration file: {config_path}")
                raise ValueError(f"Invalid JSON in configuration file: {config_path}")

        except Exception as e:
            self.logger.error(f"Error getting configuration: {str(e)}")
            raise

    def get_config_for(self, key):
        self.logger.debug(f"Getting configuration for key: {key}")
        if key not in self.config:
            self.logger.error(f"Key '{key}' not found in configuration")
            raise KeyError(f"Key '{key}' not found in configuration.")
        self.logger.debug(f"Returning config value for key '{key}': {self.config[key]}")
        return self.config[key]
