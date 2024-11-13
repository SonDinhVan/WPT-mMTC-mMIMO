"""
This class provides the set-up for nodes (i.e. AP, UE) in the network.
"""
from configs import config

from dataclasses import dataclass
import numpy as np


@dataclass
class Node:
    """
    A data class to contain nodes in the network system.

    Args:
        x (float, optional): x-coordinator position of node. Default = 0.
        y (float, optional): y-coordinator position of node. Default = 0.
    """

    x: float = 0.0
    y: float = 0.0

    def get_distance(self, node: "Node") -> float:
        """
        Get distance from itself to another node

        Args:
            node_2 (node): A node in the network

        Returns:
            float: the distance (in meters)
        """
        return np.sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2)

    def get_cos_angle(self, node: "Node") -> float:
        """
        Get the cosin value of the angle of the signal between nodes

        Args:
            node_2 (node): A node in the network

        Returns:
            float: [The cosin of angle of the signal respect to the Horizontal line]
        """
        return np.abs(self.x - node.x) / self.get_distance(node)

    def get_angle(self, node: "Node") -> float:
        """
        Get the angle of the direct signal between two nodes

        Args:
            node (Node): [A node in the network]

        Returns:
            float: [The angle of the signal respect to the Horizontal line]
        """
        return np.arccos(np.abs(self.x - node.x) / self.get_distance(node))


@dataclass
class UE(Node):
    """
    UE data class

    Args:
        P (float, P_max_ue): [Transmit power of UE]. Defaults to P_max set in config
        varphi (np.array, optional): [Pilot sequence used by UE]. Defaults to None.
        AP_index (np.array, optional): [Indices of APs serving this UE]. Defaults to None.
        pilot_index (int, optional): [Index of the used pilot]. Defaults to None.
    """

    P: float = config.P_MAX_UE
    varphi: np.array = None
    AP_index: np.array = None
    pilot_index: int = None


@dataclass
class AP(Node):
    """
    AP data class

    Args:
        N (int, optional): [the number of antenna]. Defaults to None.
        P (float, 10.0): [transmit power in Watt]. Defaults to 10.0 Watt.
        UE_index (np.array, optional): [array of indices of UEs it serves]. Defaults to None.
        w (np.array, optional): [power weights vector]. Defaults to None. This is only determined after AP selections.
    """

    N: int = None
    P: float = config.POWER_AP
    UE_index: np.array = None
    w: np.array = None

    def generate_uniform_power_weights(self):
        if len(self.UE_index) != 0:
            num_UE_served = len(self.UE_index)
            self.w = np.array(num_UE_served * [1.0 / num_UE_served]).reshape((-1, 1))

    def generate_random_power_weights(self):
        if len(self.UE_index) != 0:
            num_UE_served = len(self.UE_index)
            w = np.random.rand(num_UE_served, 1)
            self.w = w / np.sum(w)

    def steering_vector(self, theta: float = None) -> np.array:
        """
        Calculate the steering vector.

        Args:
            theta (np.float): [The AoD angle]

        Returns:
            np.array: [Steering vector]
        """
        return np.array(
            list(
                map(
                    lambda m: 1
                    / np.sqrt(self.N)
                    * np.exp(1j * np.pi * m * np.sin(theta)),
                    range(self.N),
                )
            )
        ).reshape(-1, 1)
