from typing import List

from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Settings of the application, used by workers and dashboard.
    """

    # Dash/Plotly
    debug: bool = True
    css_url: str = r'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
    app_name: str = 'FMN lab | Pyquac'
    app_logo: str = r'/assets/logo_sign.png'
    app_logo_url: str = r'https://github.com/ikaryss/pyquac/blob/main/images/logo_sign.png?raw=true'
    app_link: str = r'https://fmn.bmstu.ru/'
    app_linkedin_url: str = r'https://fmn.bmstu.ru/'
    app_github_url: str = r'https://github.com/ikaryss/pyquac'

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