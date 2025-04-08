import os
import json
from pathlib import Path
from typing import Dict, Any


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

        if self.verbose:
            print("[INFO] Initializing ConfigManager")

        self.config_dir = Path(config_dir)
        if self.verbose:
            print(f"Configuration directory set to: {self.config_dir}")

        self.mpi_config_path = mpi_config_path
        self.mpcdf_config_path = mpcdf_config_path
        if self.verbose:
            print(f"MPI config path: {mpi_config_path}")
            print(f"MPCDF config path: {mpcdf_config_path}")

        self.get_config()

    def detect_environment(self) -> str:
        """
        Detect which cluster environment is being used.

        Returns:
            str: 'mpcdf' or 'mpi' or 'unknown'
        """
        if self.verbose:
            print("Detecting environment...")

        if os.environ.get("mpcdf") is not None:
            if self.verbose:
                print("[INFO] Detected MPCDF environment")
            return "mpcdf"
        elif os.environ.get("mpi") is not None:
            if self.verbose:
                print("[INFO] Detected MPI environment")
            return "mpi"
        else:
            print("Environment not detected! Missing environment variables.")
            raise ValueError(
                "Environment not detected. Please set either 'mpcdf' or 'mpi' environment variable."
            )

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration based on detected environment.

        Returns:
            Dict[str, Any]: Configuration dictionary

        Raises:
            RuntimeError: If environment cannot be detected and no default is specified
        """
        if self.verbose:
            print("[INFO] Getting configuration based on environment")

        try:
            env = self.detect_environment()

            # Load appropriate config file
            if env == "mpcdf":
                config_path = self.config_dir / self.mpcdf_config_path
            elif env == "mpi":
                config_path = self.config_dir / self.mpi_config_path
            else:
                print(f"Unknown environment: {env}")
                raise RuntimeError(
                    "Unknown environment. Please set either 'mpcdf' or 'mpi' "
                    "environment variable, or specify a default configuration."
                )

            if self.verbose:
                print(f"[INFO] Loading configuration from: {config_path}")
            print(f"Loading configuration from: {config_path}")

            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                    if self.verbose:
                        print(f"[INFO] Configuration loaded successfully")
                    print(f"Configuration loaded successfully.")
                    if self.verbose:
                        print(f"Config contents: {self.config}")
                    return self.config
            except FileNotFoundError:
                print(f"Configuration file not found: {config_path}")
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            except json.JSONDecodeError:
                print(f"Invalid JSON in configuration file: {config_path}")
                raise ValueError(f"Invalid JSON in configuration file: {config_path}")

        except Exception as e:
            print(f"Error getting configuration: {str(e)}")
            raise

    def get_config_for(self, key):
        if self.verbose:
            print(f"Getting configuration for key: {key}")
        if key not in self.config:
            print(f"Key '{key}' not found in configuration")
            raise KeyError(f"Key '{key}' not found in configuration.")
        if self.verbose:
            print(f"Returning config value for key '{key}': {self.config[key]}")
        return self.config[key]
