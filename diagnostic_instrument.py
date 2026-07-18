#!/usr/bin/env python3
"""
DIAGNOSTIC VERSION - IoT Q-Learning Routing
Instruments every packet drop to identify root cause
"""

import numpy as np
import networkx as nx
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from collections import deque, Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration (same as before)
SIMULATION_PARAMS = {
    'num_nodes': 100,
    'area_width': 20,
    'area_height': 20,
    'tx_range': 3.5,
    'alpha': 0.7,
    'gamma': 0.9,
    'epsilon': 0.15,
    'epsilon_decay': 0.995,
    'infection_levels': [0],  # Only 0% for diagnostics
    'num_sources': 5,
    'packets_per_round': 10,
    'frames_per_infection_level': 30,  # Reduced for diagnostics
    'energy_decay_rate': 0.01,
    'max_hop_limit': 15,
}

@dataclass
class Node:
    """IoT Node representation"""
    node_id: int
    x: float
    y: float
    is_sink: bool = False
    infected: bool = False
    energy: float = 1000.0
    q_table: Dict[int, Dict[int, float]] = field(default_factory=dict)
    neighbors: Set[int] = field(default_factory=set)
    selected_next_hop: Dict[int, Optional[int]] = field(default_factory=dict)
    shortest_path_next_hop: Dict[int, Optional[int]] = field(default_factory=dict)
    
    def distance_to(self, other: 'Node') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def get_q_value(self, dest: int, next_hop: int) -> float:
        if dest not in self.q_table:
            self.q_table[dest] = {}
        return self.q_table[dest].get(next_hop, 0.0)
    
    def set_q_value(self, dest: int, next_hop: int, value: float) -> None:
        if dest not in self.q_table:
            self.q_table[dest] = {}
        self.q_table[dest][next_hop] = value
    
    def update_q_value(self, dest: int, next_hop: int, reward: float, 
                      max_next_q: float, alpha: float, gamma: float) -> float:
        current_q = self.get_q_value(dest, next_hop)
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        self.set_q_value(dest, next_hop, new_q)
        return new_q


@dataclass
class Packet:
    """Data packet for routing simulation"""
    packet_id: int
    source: int
    destination: int
    current_node: int
    path: List[int] = field(default_factory=list)
    created_time: int = 0
    delivered_time: Optional[int] = None
    hop_count: int = 0
    is_delivered: bool = False
    is_dropped: bool = False
    drop_reason: str = ""
    drop_node: int = -1  # NEW: Track where packet was dropped


class DiagnosticNetwork:
    """IoT Network with comprehensive diagnostics"""
    
    def __init__(self, params: dict):
        self.params = params
        self.nodes: Dict[int, Node] = {}
        self.graph = nx.Graph()
        self.packets: List[Packet] = []
        
        # Diagnostics
        self.drop_reasons = Counter()
        self.successful_paths = []
        self.failed_paths = []
        self.q_value_stats = {
            'max_q': 0,
            'min_q': 0,
            'mean_q': 0,
            'num_zero_q': 0,
            'num_learned_routes': 0,
        }
        self.routing_table_snapshots = []
        
        self._initialize_network()
    
    def _initialize_network(self):
        """Create network topology"""
        np.random.seed(42)
        
        # Create sink
        self.nodes[0] = Node(0, 10, 10, is_sink=True, energy=99999.0)
        self.graph.add_node(0)
        
        # Create sensors
        for i in range(1, self.params['num_nodes'] + 1):
            x = np.random.uniform(0, self.params['area_width'])
            y = np.random.uniform(0, self.params['area_height'])
            self.nodes[i] = Node(i, x, y)
            self.graph.add_node(i)
        
        # Build neighbor tables
        self._build_neighbor_tables()
        
        # Initialize shortest paths
        self._initialize_shortest_paths()
        
        # Initialize Q-tables
        self._initialize_q_tables()
        
        # Print topology
        self._print_topology_analysis()
    
    def _build_neighbor_tables(self):
        """Build neighbor tables"""
        tx_range = self.params['tx_range']
        node_list = list(self.nodes.values())
        
        for i, node_a in enumerate(node_list):
            for j, node_b in enumerate(node_list):
                if i < j:
                    distance = node_a.distance_to(node_b)
                    if distance <= tx_range:
                        node_a.neighbors.add(node_b.node_id)
                        node_b.neighbors.add(node_a.node_id)
                        self.graph.add_edge(node_a.node_id, node_b.node_id)
    
    def _initialize_shortest_paths(self):
        """Compute shortest paths via BFS"""
        distances = {0: 0}
        queue = deque([0])
        parent = {0: None}
        
        while queue:
            node_id = queue.popleft()
            node = self.nodes[node_id]
            
            for neighbor in node.neighbors:
                if neighbor not in distances:
                    distances[neighbor] = distances[node_id] + 1
                    parent[neighbor] = node_id
                    queue.append(neighbor)
        
        # Store shortest path next hop
        for node_id in range(1, self.params['num_nodes'] + 1):
            if node_id in parent and parent[node_id] is not None:
                self.nodes[node_id].shortest_path_next_hop[0] = parent[node_id]
            else:
                self.nodes[node_id].shortest_path_next_hop[0] = None
    
    def _initialize_q_tables(self):
        """Initialize Q-tables"""
        for node in self.nodes.values():
            if not node.is_sink:
                for neighbor in node.neighbors:
                    node.set_q_value(0, neighbor, 0.0)
    
    def _print_topology_analysis(self):
        """Print network topology analysis"""
        print("\n" + "=" * 80)
        print("TOPOLOGY ANALYSIS")
        print("=" * 80)
        
        components = list(nx.connected_components(self.graph))
        print(f"Connected components: {len(components)}")
        if len(components) > 1:
            print(f"  Component sizes: {[len(c) for c in components]}")
        
        # Check sink reachability
        if 0 in components[0]:
            reachable = components[0]
        else:
            reachable = set([0])
        
        print(f"Sink reachability: {len(reachable)}/{self.params['num_nodes']} nodes")
        
        # Degree distribution
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        print(f"Degree - Min: {min(degrees)}, Max: {max(degrees)}, Avg: {np.mean(degrees):.2f}")
        
        # Shortest path lengths
        lengths = nx.single_source_shortest_path_length(self.graph, 0)
        lengths_to_sink = [lengths[n] for n in range(1, self.params['num_nodes'] + 1) if n in lengths]
        if lengths_to_sink:
            print(f"Shortest paths - Min: {min(lengths_to_sink)}, Max: {max(lengths_to_sink)}, Avg: {np.mean(lengths_to_sink):.2f}")
        
        print("=" * 80 + "\n")
    
    def _select_next_hop(self, node_id: int, infected_nodes: Set[int]) -> Optional[int]:
        """Select next hop"""
        node = self.nodes[node_id]
        available = [n for n in node.neighbors if n not in infected_nodes]
        
        if not available:
            return None
        
        # Try Q-learning first
        q_values = node.q_table.get(0, {})
        valid = [n for n in available if n in q_values]
        
        if valid:
            max_q = max(q_values.get(n, 0.0) for n in valid)
            best = [n for n in valid if q_values.get(n, 0.0) == max_q]
            return np.random.choice(best) if best else None
        
        # Fallback to shortest path
        sp = node.shortest_path_next_hop.get(0)
        if sp and sp not in infected_nodes:
            return sp
        
        # Random choice if all else fails
        return np.random.choice(available) if available else None
    
    def compute_routing_tree(self, infected_nodes: Set[int]):
        """Compute routing tree"""
        for node_id in range(1, self.params['num_nodes'] + 1):
            node = self.nodes[node_id]
            next_hop = self._select_next_hop(node_id, infected_nodes)
            
            if next_hop is None:
                # Use shortest path as absolute fallback
                next_hop = node.shortest_path_next_hop.get(0)
            
            node.selected_next_hop[0] = next_hop
            
            # Simple Q-update (heuristic only)
            if next_hop is not None:
                reward = 10.0  # Simple reward
                next_node = self.nodes[next_hop]
                q_values_next = next_node.q_table.get(0, {})
                max_next_q = max(q_values_next.values()) if q_values_next else 0.0
                node.update_q_value(0, next_hop, reward, max_next_q, 
                                   self.params['alpha'], self.params['gamma'])
    
    def generate_packets(self) -> List[Packet]:
        """Generate packets"""
        packets = []
        reachable_sources = [n_id for n_id in range(1, self.params['num_nodes'] + 1)
                           if self.nodes[n_id].selected_next_hop.get(0) is not None]
        
        if not reachable_sources:
            return packets
        
        num_sources = min(self.params['num_sources'], len(reachable_sources))
        sources = np.random.choice(reachable_sources, num_sources, replace=False)
        
        for source in sources:
            for _ in range(self.params['packets_per_round']):
                packet = Packet(
                    packet_id=len(self.packets) + len(packets),
                    source=source,
                    destination=0,
                    current_node=source,
                    created_time=0,
                    path=[source]
                )
                packets.append(packet)
        
        return packets
    
    def forward_packet(self, packet: Packet, infected_nodes: Set[int]) -> bool:
        """Forward packet with detailed diagnostics"""
        current = packet.current_node
        
        # Check delivery
        if current == packet.destination:
            packet.is_delivered = True
            self.successful_paths.append(packet.path)
            return True
        
        # Check hop limit
        if packet.hop_count >= self.params['max_hop_limit']:
            packet.is_dropped = True
            packet.drop_reason = 'hop_limit'
            packet.drop_node = current
            self.drop_reasons['hop_limit'] += 1
            self.failed_paths.append((packet.path, 'hop_limit'))
            return False
        
        # Check for loops
        if current in packet.path[:-1]:
            packet.is_dropped = True
            packet.drop_reason = 'loop_detected'
            packet.drop_node = current
            self.drop_reasons['loop_detected'] += 1
            self.failed_paths.append((packet.path, 'loop'))
            return False
        
        # Get next hop
        current_node = self.nodes[current]
        next_hop = current_node.selected_next_hop.get(0)
        
        # Try fallback if no route
        if next_hop is None:
            next_hop = current_node.shortest_path_next_hop.get(0)
            if next_hop:
                self.drop_reasons['no_selected_hop_fallback_used'] += 1
        
        if next_hop is None:
            packet.is_dropped = True
            packet.drop_reason = 'no_next_hop'
            packet.drop_node = current
            self.drop_reasons['no_next_hop'] += 1
            self.failed_paths.append((packet.path, 'no_next_hop'))
            return False
        
        # Check infection
        if next_hop in infected_nodes:
            packet.is_dropped = True
            packet.drop_reason = 'infected_node'
            packet.drop_node = current
            self.drop_reasons['infected_node'] += 1
            self.failed_paths.append((packet.path, 'infected'))
            return False
        
        # Check energy
        next_node = self.nodes[next_hop]
        if next_node.energy <= 0:
            packet.is_dropped = True
            packet.drop_reason = 'energy_depleted'
            packet.drop_node = current
            self.drop_reasons['energy_depleted'] += 1
            self.failed_paths.append((packet.path, 'energy'))
            return False
        
        # Forward
        packet.current_node = next_hop
        packet.path.append(next_hop)
        packet.hop_count += 1
        next_node.energy -= self.params['energy_decay_rate']
        
        return packet.current_node != packet.destination
    
    def run_diagnostic(self):
        """Run diagnostic simulation"""
        print("\n" + "=" * 80)
        print("RUNNING DIAGNOSTIC SIMULATION")
        print("=" * 80)
        
        infected_nodes = set()  # No infection for diagnostics
        
        # Compute routing
        self.compute_routing_tree(infected_nodes)
        
        # Generate packets
        packets = self.generate_packets()
        self.packets = packets
        
        print(f"\nGenerated {len(packets)} packets")
        
        # Forward packets (up to 50 hops max)
        for hop in range(50):
            active = [p for p in packets if not p.is_delivered and not p.is_dropped]
            if not active:
                break
            
            for packet in active:
                self.forward_packet(packet, infected_nodes)
            
            print(f"Hop {hop + 1}: {len(active)} packets in transit")
        
        # Analyze results
        self._analyze_results()
    
    def _analyze_results(self):
        """Analyze simulation results"""
        print("\n" + "=" * 80)
        print("DROP REASON STATISTICS")
        print("=" * 80)
        
        total_packets = len(self.packets)
        delivered = sum(1 for p in self.packets if p.is_delivered)
        dropped = sum(1 for p in self.packets if p.is_dropped)
        
        print(f"\nTotal packets: {total_packets}")
        print(f"Delivered: {delivered} ({100*delivered/total_packets:.1f}%)")
        print(f"Dropped: {dropped} ({100*dropped/total_packets:.1f}%)")
        print(f"In-transit: {total_packets - delivered - dropped}")
        
        print("\nDrop reasons:")
        for reason, count in self.drop_reasons.most_common():
            pct = 100 * count / total_packets if total_packets > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # Path analysis
        print("\n" + "=" * 80)
        print("ROUTE ANALYSIS")
        print("=" * 80)
        
        print(f"\nSuccessful deliveries: {len(self.successful_paths)}")
        if self.successful_paths:
            path_lengths = [len(p) for p in self.successful_paths]
            print(f"  Path lengths - Min: {min(path_lengths)}, Max: {max(path_lengths)}, Avg: {np.mean(path_lengths):.2f}")
            
            print("\nSample successful paths:")
            for i, path in enumerate(self.successful_paths[:5]):
                print(f"  Path {i+1}: {' -> '.join(map(str, path))}")
        
        print(f"\nFailed deliveries: {len(self.failed_paths)}")
        if self.failed_paths:
            print("\nSample failed paths:")
            for i, (path, reason) in enumerate(self.failed_paths[:10]):
                print(f"  Path {i+1} ({reason}): {' -> '.join(map(str, path))}")
        
        # Q-learning audit
        print("\n" + "=" * 80)
        print("Q-LEARNING AUDIT")
        print("=" * 80)
        
        all_q_values = []
        learned_count = 0
        zero_q_count = 0
        max_q_values = []
        
        for node in self.nodes.values():
            if node.is_sink:
                continue
            
            q_dict = node.q_table.get(0, {})
            if q_dict:
                learned_count += 1
                values = list(q_dict.values())
                all_q_values.extend(values)
                max_q = max(values)
                max_q_values.append(max_q)
                zero_q_count += sum(1 for v in values if v == 0.0)
        
        print(f"\nNodes with learned Q-values: {learned_count}/{self.params['num_nodes']}")
        
        if all_q_values:
            print(f"Q-value statistics:")
            print(f"  Min: {min(all_q_values):.4f}")
            print(f"  Max: {max(all_q_values):.4f}")
            print(f"  Mean: {np.mean(all_q_values):.4f}")
            print(f"  Std Dev: {np.std(all_q_values):.4f}")
            print(f"  Count zeros: {zero_q_count}/{len(all_q_values)}")
            print(f"  Percent zeros: {100*zero_q_count/len(all_q_values):.1f}%")
        
        if max_q_values:
            print(f"\nMax Q-values per node - Min: {min(max_q_values):.4f}, Max: {max(max_q_values):.4f}")
        
        # Routing table analysis
        print("\n" + "=" * 80)
        print("ROUTING TABLE ANALYSIS")
        print("=" * 80)
        
        # Count how many nodes have selected_next_hop
        nodes_with_route = sum(1 for n in self.nodes.values() 
                              if not n.is_sink and n.selected_next_hop.get(0) is not None)
        print(f"\nNodes with selected next hop: {nodes_with_route}/{self.params['num_nodes']}")
        
        # Check routing loops
        print(f"\nChecking for routing loops...")
        loop_count = 0
        for node_id in range(1, self.params['num_nodes'] + 1):
            node = self.nodes[node_id]
            next_hop = node.selected_next_hop.get(0)
            if next_hop is None:
                continue
            
            # Follow the path for max 20 hops
            visited = set([node_id])
            current = next_hop
            for _ in range(20):
                if current == 0:  # Reached sink
                    break
                if current in visited:  # Loop detected
                    loop_count += 1
                    break
                visited.add(current)
                next_hop_node = self.nodes[current]
                current = next_hop_node.selected_next_hop.get(0)
                if current is None:
                    break
        
        print(f"Nodes with routing loops: {loop_count}")
        
        # Detailed routing table sample
        print("\nSample routing table:")
        for node_id in range(1, min(11, self.params['num_nodes'] + 1)):
            node = self.nodes[node_id]
            sp = node.shortest_path_next_hop.get(0)
            sq = node.selected_next_hop.get(0)
            q_vals = node.q_table.get(0, {})
            max_q = max(q_vals.values()) if q_vals else 0
            print(f"  Node {node_id}: SP={sp}, SQ={sq}, MaxQ={max_q:.2f}, Q_count={len(q_vals)}")


def main():
    """Run diagnostics"""
    network = DiagnosticNetwork(SIMULATION_PARAMS)
    network.run_diagnostic()


if __name__ == '__main__':
    main()