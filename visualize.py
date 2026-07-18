"""
Visualization utilities for IoT network topologies and routing paths.

Uses matplotlib and networkx for network visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import List, Optional, Tuple
from network import IoTNetwork, Node, TopologyType


class NetworkVisualizer:
    """Visualize IoT network topologies and routing paths."""
    
    def __init__(self, network: IoTNetwork, figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize visualizer.
        
        Args:
            network: IoTNetwork instance to visualize
            figsize: Figure size (width, height)
        """
        self.network = network
        self.figsize = figsize
    
    def _create_networkx_graph(self) -> nx.Graph:
        """
        Create NetworkX graph from IoT network.
        
        Returns:
            NetworkX Graph with nodes and edges
        """
        G = nx.Graph()
        
        # Add sensor nodes
        for node_id, node in self.network.nodes.items():
            G.add_node(node_id, pos=(node.x, node.y), type='sensor')
        
        # Add sink
        if self.network.sink:
            G.add_node(-1, pos=(self.network.sink.x, self.network.sink.y), type='sink')
        
        # Add edges (communication links)
        added_edges = set()
        for node_id, node in self.network.nodes.items():
            for neighbor_id in node.neighbor_list:
                edge = tuple(sorted([node_id, neighbor_id]))
                if edge not in added_edges:
                    G.add_edge(node_id, neighbor_id)
                    added_edges.add(edge)
        
        # Add edges from nodes to sink
        if self.network.sink:
            for node_id, node in self.network.nodes.items():
                if -1 in node.neighbor_list:
                    G.add_edge(node_id, -1)
        
        return G
    
    def plot_topology(self, output_path: Optional[str] = None, show_links: bool = True) -> None:
        """
        Plot network topology with nodes, sink, and communication links.
        
        Args:
            output_path: Path to save figure (if None, display only)
            show_links: Whether to show communication links
        """
        G = self._create_networkx_graph()
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Draw communication links (if enabled)
        if show_links:
            nx.draw_networkx_edges(
                G, pos,
                ax=ax,
                edge_color='lightgray',
                width=0.5,
                alpha=0.6
            )
        
        # Draw sensor nodes
        sensor_nodes = [n for n in G.nodes() if n != -1]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=sensor_nodes,
            ax=ax,
            node_color='lightblue',
            node_size=100,
            label='Sensor Nodes'
        )
        
        # Draw sink node
        if -1 in G.nodes():
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[-1],
                ax=ax,
                node_color='red',
                node_size=400,
                label='Sink'
            )
        
        # Draw labels
        labels = {node: str(node) if node != -1 else 'S' for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6)
        
        # Formatting
        ax.set_title(f'{self.network.topology_type.value.capitalize()} Topology\n'
                     f'Nodes: {len(self.network.nodes)}, '
                     f'Area: {self.network.area_width}×{self.network.area_height} km, '
                     f'Range: {self.network.transmission_range} km',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate (km)', fontsize=11)
        ax.set_ylabel('Y Coordinate (km)', fontsize=11)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(-1, self.network.area_width + 1)
        ax.set_ylim(-1, self.network.area_height + 1)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save or display
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Topology saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_routing_path(
        self,
        path: List[int],
        output_path: Optional[str] = None,
        show_links: bool = True
    ) -> None:
        """
        Plot network with a specific routing path highlighted.
        
        Args:
            path: List of node IDs in the route
            output_path: Path to save figure
            show_links: Whether to show all communication links
        """
        G = self._create_networkx_graph()
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Draw all communication links
        if show_links:
            nx.draw_networkx_edges(
                G, pos,
                ax=ax,
                edge_color='lightgray',
                width=0.5,
                alpha=0.3
            )
        
        # Draw sensor nodes
        sensor_nodes = [n for n in G.nodes() if n != -1]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=sensor_nodes,
            ax=ax,
            node_color='lightblue',
            node_size=80,
            alpha=0.6
        )
        
        # Highlight routing path
        path_nodes = [n for n in path if n in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=path_nodes,
            ax=ax,
            node_color='yellow',
            node_size=150,
            alpha=0.9
        )
        
        # Draw sink
        if -1 in G.nodes():
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[-1],
                ax=ax,
                node_color='red',
                node_size=300,
                alpha=0.9
            )
        
        # Draw path edges
        if len(path) > 1:
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(
                G, pos,
                edgelist=path_edges,
                ax=ax,
                edge_color='green',
                width=2.0,
                alpha=0.8,
                arrows=True,
                arrowsize=15,
                arrowstyle='->'
            )
        
        # Labels
        labels = {node: str(node) if node != -1 else 'S' for node in path_nodes + ([-1] if -1 in path_nodes else [])}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        # Formatting
        ax.set_title(f'Routing Path (Length: {len(path)-1} hops)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate (km)', fontsize=10)
        ax.set_ylabel('Y Coordinate (km)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Legend
        blue_patch = mpatches.Patch(color='lightblue', label='Sensor Nodes')
        yellow_patch = mpatches.Patch(color='yellow', label='Path Nodes')
        red_patch = mpatches.Patch(color='red', label='Sink')
        ax.legend(handles=[blue_patch, yellow_patch, red_patch], loc='upper left')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Routing path saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_energy_distribution(self, output_path: Optional[str] = None) -> None:
        """
        Plot energy levels of nodes after routing.
        
        Args:
            output_path: Path to save figure
        """
        G = self._create_networkx_graph()
        pos = nx.get_node_attributes(G, 'pos')
        
        # Get energy values for coloring
        node_energies = []
        node_ids = []
        for node_id in self.network.nodes.keys():
            node = self.network.nodes[node_id]
            node_energies.append(node.energy_ratio())
            node_ids.append(node_id)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=0.5, alpha=0.4)
        
        # Draw nodes colored by energy
        nodes_collection = nx.draw_networkx_nodes(
            G, pos,
            nodelist=node_ids,
            ax=ax,
            node_color=node_energies,
            node_size=150,
            cmap=plt.cm.RdYlGn,
            vmin=0,
            vmax=1
        )
        
        # Draw sink
        if -1 in G.nodes():
            nx.draw_networkx_nodes(G, pos, nodelist=[-1], ax=ax, node_color='blue', node_size=400)
        
        # Colorbar
        cbar = plt.colorbar(nodes_collection, ax=ax, label='Energy Ratio')
        
        ax.set_title(f'Node Energy Distribution\n'
                     f'Green=High, Red=Low', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate (km)', fontsize=10)
        ax.set_ylabel('Y Coordinate (km)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Energy distribution saved: {output_path}")
        else:
            plt.show()
        
        plt.close()


def visualize_all_topologies(output_dir: str = "outputs") -> None:
    """
    Create and visualize all three topologies.
    
    Args:
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    topologies = [
        (TopologyType.RANDOM, f"{output_dir}/random_topology.png"),
        (TopologyType.MESH, f"{output_dir}/mesh_topology.png"),
        (TopologyType.TREE, f"{output_dir}/tree_topology.png"),
    ]
    
    for topo_type, output_path in topologies:
        print(f"\nGenerating {topo_type.value} topology...")
        network = IoTNetwork(topo_type)
        visualizer = NetworkVisualizer(network)
        visualizer.plot_topology(output_path)
        
        # Print statistics
        stats = network.get_statistics()
        print(f"  Nodes: {stats['total_nodes']}")
        print(f"  Transmission Range: {network.transmission_range} km")
        print(f"  Area: {network.area_width}×{network.area_height} km")
