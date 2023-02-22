from typing import List

from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Settings of the application, used by workers and dashboard.
    """

    # Dash/Plotly
    debug: bool = True
    css_url: str = r"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
    app_name: str = "fmn | pyquac"
    app_logo: str = r"/assets/logo_sign.png"
    app_logo_url: str = (
        r"https://github.com/ikaryss/pyquac/blob/main/images/logo_sign.png?raw=true"
    )
    app_logo_pyquac: str = r"assets/logo_pyquac.png"
    app_link: str = r"https://fmn.bmstu.ru/"
    app_linkedin_url: str = r"https://fmn.bmstu.ru/"
    app_github_url: str = r"https://github.com/ikaryss/pyquac"
    transition_time: str = "all 0.5s"
    margin_transition_time: str = "margin-left .5s"

    # App settings
    init_interval: int = 3000
    init_max_interval: int = -1
    init_x_label: str = "Voltages, V"
    init_y_label: str = "Frequencies, GHz"
    init_cmap: str = "rdylbu"
    init_xy_lines_state: bool = False

    # Saving settings
    default_root_path: str = r"D:/Scripts/Measurement_automation/data/qubits/"

    class Config:
        """
        Meta configuration of the settings parser
        """

        env_file = ".env"
        # Prefix the environment variable not to mix up with other variables
        # used by the OS or other software.

        env_prefix = "SMD_"
        # SMD stands for Stock Market Dashboard


settings = Settings()
