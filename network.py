"""
Network topology implementation for IoT Mesh Networks
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple
import math
import random
from enum import Enum


class TopologyType(Enum):
    """Supported topology types as per paper."""
    RANDOM = "random"
    MESH = "mesh"
    TREE = "tree"


@dataclass
class Node:
    """
    Represents a node in the IoT network.
    
    Attributes:
        node_id: Unique identifier
        x: X coordinate (km)
        y: Y coordinate (km)
        initial_energy: Initial energy in Joules (from paper: 90 J)
        residual_energy: Current energy in Joules
        packet_sent: Count of packets sent
        packet_received: Count of packets received
        neighbor_list: Set of neighbor node IDs
    """
    node_id: int
    x: float
    y: float
    initial_energy: float = 90.0  # As per paper
    residual_energy: float = 90.0
    packet_sent: int = 0
    packet_received: int = 0
    neighbor_list: Set[int] = field(default_factory=set)
    
    def reset_traffic(self) -> None:
        """Reset traffic counters without affecting energy."""
        self.packet_sent = 0
        self.packet_received = 0
    
    def update_energy(self, energy_consumed: float) -> None:
        """Update residual energy after packet transmission."""
        self.residual_energy = max(0, self.residual_energy - energy_consumed)
    
    def energy_ratio(self) -> float:
        """Return residual_energy / initial_energy for reward calculation."""
        if self.initial_energy == 0:
            return 0.0
        return self.residual_energy / self.initial_energy


class IoTNetwork:
    """
    IoT Mesh Network implementation with three topologies.
    
    Topologies from paper:
    1. Random: 100 nodes, 20×20 km, 5 km range
    2. Mesh: 38 nodes, 25×25 km, 4.5 km range
    3. Tree: 21 nodes, 50×50 km, 4 km range
    """
    
    def __init__(self, topology_type: TopologyType = TopologyType.MESH):
        """
        Initialize network with specified topology.
        
        Args:
            topology_type: One of RANDOM, MESH, or TREE
        """
        self.topology_type = topology_type
        self.nodes: dict[int, Node] = {}
        self.sink: Node | None = None
        self.transmission_range: float = 0.0
        self.area_width: float = 0.0
        self.area_height: float = 0.0
        
        # Generate topology
        if topology_type == TopologyType.RANDOM:
            self._generate_random_topology()
        elif topology_type == TopologyType.MESH:
            self._generate_mesh_topology()
        elif topology_type == TopologyType.TREE:
            self._generate_tree_topology()
        
        # Discover neighbors for all nodes
        self._discover_neighbors()
    
    def _generate_random_topology(self) -> None:
        """
        Generate Random topology: 100 nodes, 20×20 km, 5 km transmission range.
        Sink at bottom region (y < 5).
        """
        self.area_width = 20.0
        self.area_height = 20.0
        self.transmission_range = 5.0
        
        # Create 100 nodes
        for i in range(100):
            x = random.uniform(0, self.area_width)
            y = random.uniform(0, self.area_height)
            node = Node(node_id=i, x=x, y=y)
            self.nodes[i] = node
        
        # Place sink in bottom region (y < 5)
        sink_x = random.uniform(0, self.area_width)
        sink_y = random.uniform(0, 5)
        self.sink = Node(node_id=-1, x=sink_x, y=sink_y)
    
    def _generate_mesh_topology(self) -> None:
        """
        Generate Mesh topology: 38 nodes, 25×25 km, 4.5 km transmission range.
        Sink at center (12.5, 12.5).
        """
        self.area_width = 25.0
        self.area_height = 25.0
        self.transmission_range = 4.5
        
        # Create 38 nodes in uniform grid pattern
        grid_size = 7  # ~7x6 grid for 38 nodes
        node_id = 0
        for i in range(grid_size):
            for j in range(7):
                if node_id >= 38:
                    break
                x = (i + 0.5) * (self.area_width / grid_size)
                y = (j + 0.5) * (self.area_height / grid_size)
                # Add small random perturbation
                x += random.uniform(-0.5, 0.5)
                y += random.uniform(-0.5, 0.5)
                x = max(0, min(self.area_width, x))
                y = max(0, min(self.area_height, y))
                node = Node(node_id=node_id, x=x, y=y)
                self.nodes[node_id] = node
                node_id += 1
        
        # Place sink at center
        self.sink = Node(node_id=-1, x=12.5, y=12.5)
    
    def _generate_tree_topology(self) -> None:
        """
        Generate Tree topology: 21 nodes, 50×50 km, 4 km transmission range.
        Sink at top-center (25, 0).
        """
        self.area_width = 50.0
        self.area_height = 50.0
        self.transmission_range = 4.0
        
        # Create 21 nodes in tree-like structure
        node_id = 0
        layers = 4
        nodes_per_layer = [1, 4, 8, 8]
        y_positions = [10, 20, 30, 40]
        
        for layer in range(layers):
            if node_id >= 21:
                break
            y = y_positions[layer]
            for pos in range(nodes_per_layer[layer]):
                if node_id >= 21:
                    break
                x = (pos + 1) * (self.area_width / (nodes_per_layer[layer] + 1))
                # Add random perturbation
                x += random.uniform(-1, 1)
                y_pos = y + random.uniform(-1, 1)
                x = max(0, min(self.area_width, x))
                y_pos = max(0, min(self.area_height, y_pos))
                node = Node(node_id=node_id, x=x, y=y_pos)
                self.nodes[node_id] = node
                node_id += 1
        
        # Place sink at top-center
        self.sink = Node(node_id=-1, x=25.0, y=0.0)
    
    def _discover_neighbors(self) -> None:
        """
        Discover neighbors for all nodes using Euclidean distance.
        
        Equation (8) & (9): Node j is neighbor of node i if distance <= transmission_range.
        """
        all_nodes = list(self.nodes.values()) + ([self.sink] if self.sink else [])
        
        for i, node_i in enumerate(all_nodes):
            for node_j in all_nodes[i + 1:]:
                dist = self.euclidean_distance(node_i, node_j)
                if dist <= self.transmission_range:
                    node_i.neighbor_list.add(node_j.node_id)
                    node_j.neighbor_list.add(node_i.node_id)
    
    @staticmethod
    def euclidean_distance(node1: Node, node2: Node) -> float:
        """
        Calculate Euclidean distance between two nodes.
        
        Distance = sqrt((x1-x2)² + (y1-y2)²)
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Distance in km
        """
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
    
    def find_neighbors(self, node_id: int) -> Set[int]:
        """
        Get neighbor list for a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Set of neighbor node IDs (does not include sink)
        """
        if node_id in self.nodes:
            return self.nodes[node_id].neighbor_list.copy()
        return set()
    
    def get_node(self, node_id: int) -> Node | None:
        """Get node by ID."""
        if node_id == -1:
            return self.sink
        return self.nodes.get(node_id)
    
    def get_all_nodes(self) -> List[Node]:
        """Get all nodes including sink."""
        nodes = list(self.nodes.values())
        if self.sink:
            nodes.append(self.sink)
        return nodes
    
    def reset_traffic_counters(self) -> None:
        """Reset all packet counters."""
        for node in self.nodes.values():
            node.reset_traffic()
        if self.sink:
            self.sink.reset_traffic()
    
    def get_statistics(self) -> dict:
        """
        Compute network statistics.
        
        Returns:
            Dictionary with network metrics
        """
        total_energy = sum(node.residual_energy for node in self.nodes.values())
        total_packets_sent = sum(node.packet_sent for node in self.nodes.values())
        total_packets_received = sum(node.packet_received for node in self.nodes.values())
        
        return {
            "total_nodes": len(self.nodes),
            "total_energy": total_energy,
            "avg_energy_per_node": total_energy / len(self.nodes) if self.nodes else 0,
            "total_packets_sent": total_packets_sent,
            "total_packets_received": total_packets_received,
            "topology": self.topology_type.value,
        }
