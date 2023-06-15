# notes
import socket
import numpy as np
from .settings import settings


def is_port_in_use(port: int) -> bool:
    """checks if selected port is in use

    Args:
        port (int): port from localhost

    Returns:
        bool: true if port is in use else false
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def initial_spectroscopy_in_app(spectroscopy):
    """_summary_

    Args:
        spectroscopy (list or dict): _description_

    Returns:
        list: initial_data, initial_chip, initial_qubittoggle, initial_type, db_disabled
    """
    bd_columns = set(
        [
            settings.data_column,
            settings.chip_column,
            settings.qubit_column,
            settings.spectroscopy_type_column,
        ]
    )

    # spectroscopy is a dict
    if isinstance(spectroscopy, dict) is True:
        data = spectroscopy[settings.data_column]
        chip = spectroscopy[settings.chip_column]
        qubit = spectroscopy[settings.qubit_column]
        spectroscopy_type = spectroscopy[settings.spectroscopy_type_column]

        if (set(spectroscopy.keys()) != bd_columns) or (
            not dict_spectroscopy_fits_well(spectroscopy_type)
        ):
            raise AttributeError
        return (
            data,
            chip,
            qubit,
            spectroscopy_type,
        )
    # spectroscopy is a list
    data = [_[settings.data_column] for _ in spectroscopy]
    chip = [_[settings.chip_column] for _ in spectroscopy]
    qubit = [_[settings.qubit_column] for _ in spectroscopy]
    spectroscopy_type = [_[settings.spectroscopy_type_column] for _ in spectroscopy]
    if (set(spectroscopy[0].keys()) != bd_columns) or (
        not list_spectroscopy_fits_well(spectroscopy_type)
    ):
        raise AttributeError
    return (
        data,
        chip,
        qubit,
        spectroscopy_type,
    )


def dict_spectroscopy_fits_well(spectroscopy_type: str):
    return spectroscopy_type.upper() in settings.allowed_types


def list_spectroscopy_fits_well(spectroscopy_type: list):
    spectroscopy_types = [type_.upper() for type_ in spectroscopy_type]
    return all(np.isin(spectroscopy_types, settings.allowed_types))
