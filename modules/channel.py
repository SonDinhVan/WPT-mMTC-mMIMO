"""
This class provides useful functions for generating the channel coefficients
, i.e. Path Loss, small-scale, shadowing.
...
"""
from configs import config
from modules import node

import numpy as np


def calculate_K(d: float) -> float:
    """
    Calculate the Rician K factor based on the distance d (m)

    Args:
        d (float): distance in m

    Returns:
        float: K
    """

    k_subsce = -config.D_CLUTTER / np.log(1 - config.R_DENS)
    p_los = np.exp(-d / k_subsce)

    return p_los / (1 - p_los)


def calculate_NLOS_pathloss(tx: node.Node, rx: node.Node) -> float:
    """
    Calculate the path-loss in dB based on Indoor Factory model by ETSI TR 138 901 V16.1.0 (2020-11).
    Link: https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/16.01.00_60/tr_138901v160100p.pdf
    This only considers the NLOS channel.

    Args:
        TX: Transmitter
        RX: Reiceiver

    Returns:
        float: Path-loss, not in dB
    """
    d = tx.get_distance(rx)
    # Shadowing for NLOS channel
    z = 5.7 * np.random.normal(0, 1)
    # PL in dB
    PL = z + 33.0 + 25.5 * np.log10(d) + 20.0 * np.log10(config.FREQUENCY)

    return 10 ** (-PL / 10)


def generate_h(tx: node.AP, rx: node.Node) -> np.array:
    """
    Generate a small scale channel h following Rayleigh distribution

    Args:
        tx (node.AP): The AP
        rx (node.Node): The UE

    Returns:
        channel array
    """
    return (
        np.random.normal(loc=0.0, scale=1.0, size=(tx.N, 1))
        + 1j * np.random.normal(loc=0.0, scale=1.0, size=(tx.N, 1))
    ) / np.sqrt(2)
