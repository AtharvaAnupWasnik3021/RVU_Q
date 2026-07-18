#!/usr/bin/env python3
"""
 Trust-Aware Geographic Routing with Q-Learning


"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from collections import deque, Counter
import warnings
import os
from trust import TrustManager, TRUST_WEIGHT, THREAT_PENALTY, TRUST_THRESHOLD

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SIMULATION_PARAMS = {
    'num_nodes': 100,
    'area_width': 20,
    'area_height': 20,
    'tx_range': 3.5,
    'alpha': 0.3,           # Q-learning rate (conservative)
    'gamma': 0.9,           # Discount factor
    'infection_levels': [0, 10, 20, 30, 40, 50, 60],
    'num_sources': 5,
    'packets_per_round': 10,
    'frames_per_infection_level': 120,
    'animation_fps': 30,
    'energy_decay_rate': 0.01,
    'max_hop_limit': 25,
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Node:
    """IoT Node with minimal routing state"""
    node_id: int
    x: float
    y: float
    is_sink: bool = False
    infected: bool = False
    energy: float = 1000.0

    def distance_to(self, other: "Node") -> float:
        """Return Euclidean distance to another node."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    # Q-learning: [distance_class][neighbor_id] = q_value 
    # Only used as tiebreaker for equal-distance neighbors
    q_table: Dict[int, Dict[int, float]] = field(default_factory=dict)

    # Node topology
    neighbors: Set[int] = field(default_factory=set)

    # Statistics
    packets_forwarded: int = 0
    packets_dropped_here: int = 0
    route_changes: int = 0

    # Trust-related
    trust_score: float = 1.0
    packets_sent: int = 0
    packets_delivered: int = 0
    packets_dropped: int = 0
    behaviour_score: float = 1.0
    initial_energy: float = 1000.0


@dataclass
class Packet:
    """Data packet with path tracking"""
    packet_id: int
    source: int
    destination: int
    current_node: int
    path: List[int] = field(default_factory=list)
    visited_nodes: Set[int] = field(default_factory=set)
    created_time: int = 0
    delivered_time: Optional[int] = None
    hop_count: int = 0
    is_delivered: bool = False
    is_dropped: bool = False
    drop_reason: str = ""
    
    def __post_init__(self):
        """Initialize path and visited tracking"""
        if not self.path:
            self.path = [self.source]
        if not self.visited_nodes:
            self.visited_nodes = {self.source}


@dataclass
class SimulationMetrics:
    """Metrics tracking"""
    infection_percentage: List[float] = field(default_factory=list)
    pdr: List[float] = field(default_factory=list)
    avg_delay: List[float] = field(default_factory=list)
    avg_hop_count: List[float] = field(default_factory=list)
    connectivity: List[float] = field(default_factory=list)
    residual_energy: List[float] = field(default_factory=list)
    connected_nodes: List[int] = field(default_factory=list)
    delivered_packets: List[int] = field(default_factory=list)
    dropped_packets: List[int] = field(default_factory=list)
    avg_reward: List[float] = field(default_factory=list)
    avg_trust: List[float] = field(default_factory=list)


# ============================================================================
# GEOGRAPHIC ROUTING ENGINE - REFACTORED
# ============================================================================

class GeographicRoutingEngine:
    """
    Simplified greedy geographic routing with Q-learning tiebreaker.
    
    Routing priority (STRICT):
    1. Distance to sink (DOMINANT - 60%)
    2. Trust score (20%)
    3. Residual energy (10%)
    4. Link quality (10%)
    5. Q-learning (tiebreaker only, for equal distance neighbors)
    
    Key: Q-learning CANNOT override distance constraints.
    Q-learning can ONLY choose among equal-distance valid neighbors.
    """
    
    def __init__(self, network):
        self.network = network
        self.sink_id = 0
    
    def distance_to_sink(self, node_id: int) -> float:
        """Euclidean distance from node to sink"""
        node = self.network.nodes[node_id]
        sink = self.network.nodes[self.sink_id]
        return np.sqrt((node.x - sink.x)**2 + (node.y - sink.y)**2)
    
    def select_next_hop(self, current_id: int, infected_nodes: Set[int],
                       visited_nodes: Set[int], debug: bool = False) -> Optional[int]:
        """
        Select next hop for packet routing.
        
        ROUTING PRIORITY (STRICT):
        1. Must move closer to sink (distance decreases)
        2. Must not be visited (prevent loops)
        3. Must not be infected
        4. Among valid neighbors, prefer by: trust > energy > link_quality
        5. Q-learning as tiebreaker only
        
        Args:
            current_id: Current node
            infected_nodes: Set of infected node IDs
            visited_nodes: Nodes already visited by this packet
            debug: Print routing decision
            
        Returns:
            Next hop node ID, or None if no valid neighbor
        """
        current_node = self.network.nodes[current_id]
        current_dist = self.distance_to_sink(current_id)
        sink_dist = self.distance_to_sink(self.sink_id)
        
        # STEP 1: Filter neighbors by fundamental constraints
        valid_neighbors = []
        for neighbor_id in current_node.neighbors:
            neighbor = self.network.nodes[neighbor_id]
            neighbor_dist = self.distance_to_sink(neighbor_id)
            
            # Must move closer to sink
            if neighbor_dist >= current_dist:
                continue
            
            # Must not be visited (loop prevention)
            if neighbor_id in visited_nodes:
                continue
            
            # Must not be infected
            if neighbor_id in infected_nodes:
                continue
            
            # This neighbor is valid
            valid_neighbors.append(neighbor_id)
        
        if not valid_neighbors:
            return None
        
        # STEP 2: Among valid neighbors, compute scores
        scores = {}
        for neighbor_id in valid_neighbors:
            neighbor = self.network.nodes[neighbor_id]
            neighbor_dist = self.distance_to_sink(neighbor_id)
            
            # Multi-objective scoring (distance is primary)
            distance_progress = current_dist - neighbor_dist
            trust = neighbor.trust_score
            energy = neighbor.energy / neighbor.initial_energy
            
            link_dist = current_node.distance_to(neighbor)
            link_quality = max(0, 1 - link_dist / SIMULATION_PARAMS['tx_range'])
            
            # Weighted score (distance dominant)
            score = (
                0.60 * distance_progress +    # Distance progress (PRIMARY)
                0.20 * trust +                # Trust
                0.10 * energy +               # Energy
                0.10 * link_quality           # Link quality
            )
            
            scores[neighbor_id] = score
        
        # STEP 3: Select best neighbor
        best_neighbors = [n for n, s in scores.items() 
                         if s == max(scores.values())]
        
        # Tiebreaker: If multiple equally good neighbors, use Q-learning
        if len(best_neighbors) > 1:
            next_hop = self._select_by_qlearning(current_id, best_neighbors)
        else:
            next_hop = best_neighbors[0]
        
        if debug:
            print(f"  Node {current_id} → {next_hop} "
                  f"(dist {current_dist:.1f} → {self.distance_to_sink(next_hop):.1f})")
        
        return next_hop
    
    def _select_by_qlearning(self, current_id: int, 
                            neighbors: List[int]) -> int:
        """Select among equal-score neighbors using Q-learning"""
        node = self.network.nodes[current_id]
        
        # Get Q-values for neighbors
        q_values = {}
        for neighbor_id in neighbors:
            q_value = node.q_table.get(0, {}).get(neighbor_id, 0.0)
            q_values[neighbor_id] = q_value
        
        # Pick highest Q-value
        best = max(neighbors, key=lambda n: q_values.get(n, 0.0))
        return best
    
    def get_shortest_path_first_hop(self, current_id: int, 
                                    infected_nodes: Set[int],
                                    visited_nodes: Set[int],
                                    sink_id: int = 0) -> Optional[int]:
        """
        Get first hop of shortest path from current node to sink.
        Used as recovery when greedy routing has no valid neighbor.
        
        Compute on-demand using NetworkX BFS.
        """
        try:
            # Temporary graph excluding infected nodes
            temp_graph = self.network.graph.copy()
            # Remove previously visited nodes (except the current node) to prevent loops.
            for visited_id in visited_nodes:
                if visited_id != current_id and temp_graph.has_node(visited_id):
                    temp_graph.remove_node(visited_id)
            for infected_id in infected_nodes:
                if infected_id != sink_id:
                    temp_graph.remove_node(infected_id)
            
            if current_id not in temp_graph or sink_id not in temp_graph:
                return None
            
            # Find shortest path
            path = nx.shortest_path(temp_graph, current_id, sink_id)
            if len(path) <= 1:
                return None

            first_hop = path[1]

            # Ensure recovery returns a direct neighbor.
            if first_hop not in self.network.nodes[current_id].neighbors:
                return None

            return first_hop
        except nx.NetworkXNoPath:
            return None
        except Exception:
            return None


# ============================================================================
# MAIN SIMULATION - REFACTORED
# ============================================================================

class IoTMeshNetwork:
    """IoT mesh network with dynamic per-packet geographic routing"""
    
    def __init__(self, params: dict, debug: bool = False):
        self.params = params
        self.nodes: Dict[int, Node] = {}
        self.graph = nx.Graph()
        self.packets: List[Packet] = []
        self.packets_archive: List[Packet] = []
        self.metrics = SimulationMetrics()
        self.current_time = 0
        self.q_updates = []
        self.drop_stats = Counter()
        self.debug = debug
        
        self.trust_manager = TrustManager(num_nodes=params['num_nodes'])
        self.routing_engine = None
        
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize network topology"""
        np.random.seed(42)
        
        # Create sink
        self.nodes[0] = Node(0, 10, 10, is_sink=True, 
                            energy=999999.0, initial_energy=999999.0)
        self.graph.add_node(0)
        
        # Create sensor nodes
        for i in range(1, self.params['num_nodes'] + 1):
            x = np.random.uniform(0, self.params['area_width'])
            y = np.random.uniform(0, self.params['area_height'])
            node = Node(i, x, y)
            node.initial_energy = node.energy
            self.nodes[i] = node
            self.graph.add_node(i)
        
        # Build neighborhood
        self._build_neighbor_tables()
        
        # Initialize Q-tables (for tiebreaker use)
        self._initialize_q_tables()
        
        # Create routing engine
        self.routing_engine = GeographicRoutingEngine(self)
        
        self._print_diagnostics()
    
    def _build_neighbor_tables(self):
        """Build neighborhood based on transmission range"""
        tx_range = self.params['tx_range']
        nodes_list = list(self.nodes.values())
        
        for i, node_a in enumerate(nodes_list):
            for j, node_b in enumerate(nodes_list):
                if i < j:
                    distance = node_a.distance_to(node_b)
                    if distance <= tx_range:
                        node_a.neighbors.add(node_b.node_id)
                        node_b.neighbors.add(node_a.node_id)
                        self.graph.add_edge(node_a.node_id, node_b.node_id)
    
    def _initialize_q_tables(self):
        """Initialize Q-tables for all nodes (used as tiebreaker)"""
        for node in self.nodes.values():
            if not node.is_sink:
                node.q_table[0] = {}
                for neighbor_id in node.neighbors:
                    node.q_table[0][neighbor_id] = 0.0
    
    def _print_diagnostics(self):
        """Print network diagnostics"""
        components = list(nx.connected_components(self.graph))
        reachable = len(components[0]) if 0 in components[0] else 1
        
        print("=" * 80)
        print("NETWORK DIAGNOSTICS (REFACTORED ROUTING)")
        print("=" * 80)
        print(f"Nodes: {self.params['num_nodes']}")
        print(f"Edges: {self.graph.number_of_edges()}")
        print(f"Avg degree: {2*self.graph.number_of_edges()/self.params['num_nodes']:.2f}")
        print(f"Connected components: {len(components)}")
        print(f"Sink reachability: {reachable}/{self.params['num_nodes']} nodes")
        print(f"Routing: Dynamic per-packet geographic + Q-learning")
        print(f"Priority: Distance (60%) > Trust (20%) > Energy (10%) > Link (10%)")
        print("=" * 80 + "\n")
    
    def generate_packets(self, infected_nodes: Set[int]) -> List[Packet]:
        """
        Generate packets from all healthy (non-infected) non-sink nodes, regardless of routing feasibility.
        Packet generation is independent of routing feasibility; routing success or failure is determined by the routing engine.
        """
        packets = []

        # Every healthy non-sink node is eligible to generate traffic.
        healthy_sources = [
            node_id
            for node_id in range(1, self.params['num_nodes'] + 1)
            if node_id not in infected_nodes
        ]

        if not healthy_sources:
            return packets

        num_sources = min(self.params['num_sources'], len(healthy_sources))
        selected_sources = np.random.choice(
            healthy_sources,
            num_sources,
            replace=False
        )

        for source in selected_sources:
            self.nodes[source].packets_sent += self.params['packets_per_round']

            for _ in range(self.params['packets_per_round']):
                packet = Packet(
                    packet_id=len(self.packets_archive) + len(self.packets) + len(packets),
                    source=source,
                    destination=0,
                    current_node=source,
                    created_time=self.current_time
                )
                packets.append(packet)

        return packets
    
    def forward_packet(self, packet: Packet, infected_nodes: Set[int]) -> bool:
        """
        Forward packet ONE HOP using dynamic geographic routing.
        
        REFACTORED: No static routing tree. Each packet makes its own decision.
        
        Process:
        1. Check if at sink → deliver
        2. Check hop limit → drop if exceeded
        3. Select next hop using greedy geographic routing
        4. If greedy fails, try shortest path recovery
        5. If both fail, drop as void
        6. Check if next hop is infected → drop
        7. Forward and continue
        """
        
        # Check hop limit
        if packet.hop_count >= self.params['max_hop_limit']:
            packet.is_dropped = True
            packet.drop_reason = 'hoplimit'
            self.nodes[packet.source].packets_dropped += 1
            # Immediate trust penalty.
            self.nodes[packet.source].behaviour_score = max(
                0.0,
                self.nodes[packet.source].behaviour_score - 0.03
            )
            # TrustManager is responsible for trust_score calculation.
            self.drop_stats['hoplimit'] += 1
            return False
        
        
        # STEP 1: Try greedy geographic routing
        next_hop = self.routing_engine.select_next_hop(
            packet.current_node, infected_nodes, packet.visited_nodes,
            debug=self.debug
        )
        # Loop safety: reject any next hop that was already visited.
        # The current node is always in visited_nodes, so checking current_node
        # itself incorrectly drops every packet on its first forwarding step.
        if next_hop is not None and next_hop in packet.visited_nodes:
            packet.is_dropped = True
            packet.drop_reason = 'loop'
            self.nodes[packet.source].packets_dropped += 1
            # Immediate trust penalty.
            self.nodes[packet.source].behaviour_score = max(
                0.0,
                self.nodes[packet.source].behaviour_score - 0.03
            )
            # TrustManager is responsible for trust_score calculation.
            self.drop_stats['loop'] += 1
            return False
        
        # STEP 2: If greedy fails, try shortest path recovery
        if next_hop is None:
            next_hop = self.routing_engine.get_shortest_path_first_hop(
                packet.current_node,
                infected_nodes,
                packet.visited_nodes
            )
            # Recovery path must also respect loop prevention.
            if next_hop in packet.visited_nodes:
                next_hop = None
            if next_hop is not None:
                self.drop_stats['void_recovery'] += 1
        
        # STEP 3: If all routing fails, drop as void
        if next_hop is None:
            packet.is_dropped = True
            packet.drop_reason = 'void'
            self.nodes[packet.source].packets_dropped += 1
            # Immediate trust penalty.
            self.nodes[packet.source].behaviour_score = max(
                0.0,
                self.nodes[packet.source].behaviour_score - 0.03
            )
            # TrustManager is responsible for trust_score calculation.
            self.drop_stats['void'] += 1
            # Update Q-values with drop penalty
            self._update_qlearning_path(packet.path, reward=-5.0)
            return False
        
        # Basic routing validation.
        if next_hop is not None and next_hop not in self.nodes:
            raise AssertionError(f"Invalid next hop: {next_hop}")

        # Validation assertions (catch bugs)
        if next_hop == packet.current_node:
            raise AssertionError("Next hop == current node!")

        if next_hop in packet.visited_nodes:
            packet.is_dropped = True
            packet.drop_reason = 'loop'
            self.nodes[packet.source].packets_dropped += 1
            # Immediate trust penalty.
            self.nodes[packet.source].behaviour_score = max(
                0.0,
                self.nodes[packet.source].behaviour_score - 0.03
            )
            # TrustManager is responsible for trust_score calculation.
            self.drop_stats['loop'] += 1
            self._update_qlearning_path(packet.path, reward=-5.0)
            return False

        if next_hop not in self.nodes[packet.current_node].neighbors:
            raise AssertionError("Not a neighbor!")

        # Geographic progress validation. Recovery paths are allowed to make
        # non-greedy moves, but greedy forwarding should never move farther
        # from the sink.
        if next_hop is not None:
            current_dist = self.routing_engine.distance_to_sink(packet.current_node)
            next_dist = self.routing_engine.distance_to_sink(next_hop)
            if next_hop in self.nodes[packet.current_node].neighbors and next_dist > current_dist:
                if self.debug:
                    print(f"Warning: recovery selected non-greedy hop {packet.current_node}->{next_hop}")
        
        # STEP 4: Check if next hop is infected
        if next_hop in infected_nodes:
            packet.is_dropped = True
            packet.drop_reason = 'infected'
            self.nodes[packet.source].packets_dropped += 1
            # Immediate trust penalty.
            self.nodes[packet.source].behaviour_score = max(
                0.0,
                self.nodes[packet.source].behaviour_score - 0.03
            )
            # TrustManager is responsible for trust_score calculation.
            self.drop_stats['infected'] += 1
            # Update Q-values with infected penalty
            self._update_qlearning_path(packet.path, reward=-10.0)
            return False
        
        # STEP 5: Forward packet
        packet.current_node = next_hop
        # After updating current_node, increment route_changes for this node
        self.nodes[packet.current_node].route_changes += 1
        packet.path.append(next_hop)
        packet.visited_nodes.add(next_hop)
        packet.hop_count += 1
        if packet.hop_count != len(packet.path) - 1:
            raise AssertionError("Hop count and path length are inconsistent")

        # Deliver immediately upon reaching the sink.
        if packet.current_node == packet.destination == 0:
            packet.is_delivered = True
            packet.delivered_time = self.current_time
            self.nodes[packet.source].packets_delivered += 1
            self.nodes[packet.source].behaviour_score = min(
                1.0,
                self.nodes[packet.source].behaviour_score + 0.02
            )
            self._update_qlearning_path(packet.path, reward=20.0)
            return True

        next_node = self.nodes[next_hop]
        next_node.energy -= self.params['energy_decay_rate']
        next_node.packets_forwarded += 1
        # Reward forwarding behaviour.
        next_node.behaviour_score = min(1.0, next_node.behaviour_score + 0.005)
        # TrustManager is responsible for trust_score calculation.

        # Check energy
        if next_node.energy <= 0:
            packet.is_dropped = True
            packet.drop_reason = 'energy'
            self.nodes[packet.source].packets_dropped += 1
            # Immediate trust penalty.
            self.nodes[packet.source].behaviour_score = max(
                0.0,
                self.nodes[packet.source].behaviour_score - 0.03
            )
            # TrustManager is responsible for trust_score calculation.
            self.drop_stats['energy'] += 1
            return False

        return True
    
    def _update_qlearning_path(self, path: List[int], reward: float):
        """
        Update Q-values for packet path using actual delivery/drop outcome.
        
        This is CONSEQUENTIALIST Q-learning:
        - Only update when packet delivered or definitively dropped
        - Use actual observed reward, not heuristic
        - Credit entire path equally
        """
        alpha = self.params['alpha']
        gamma = self.params['gamma']
        
        for i in range(len(path) - 1):
            current_id = path[i]
            next_id = path[i + 1]
            node = self.nodes[current_id]
            
            if 0 not in node.q_table:
                node.q_table[0] = {}
            
            # Get current Q-value
            current_q = node.q_table[0].get(next_id, 0.0)
            
            # Get max Q' for next state
            next_node = self.nodes[next_id]
            next_q_values = next_node.q_table.get(0, {})
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
            
            # Q-learning update
            new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
            clipped_q = float(np.clip(new_q, -100, 100))
            node.q_table[0][next_id] = clipped_q

            # Record update for convergence analysis and metrics.
            self.q_updates.append({
                'timestep': self.current_time,
                'node': current_id,
                'neighbor': next_id,
                'old_q': float(current_q),
                'new_q': clipped_q,
                'reward': float(reward),
                'td_error': float(reward + gamma * max_next_q - current_q)
            })

        # Prevent unbounded memory growth while preserving recent learning history.
        if len(self.q_updates) > 50000:
            self.q_updates = self.q_updates[-25000:]
    
    def introduce_infection(self, percentage: int):
        """Mark random nodes as infected"""
        num_infected = int(percentage / 100.0 * self.params['num_nodes'])
        
        for node in self.nodes.values():
            if not node.is_sink:
                node.infected = False
        
        candidates = [n for n in self.nodes.keys() if not self.nodes[n].is_sink]
        if num_infected > 0 and candidates:
            infected = np.random.choice(candidates, 
                                       min(num_infected, len(candidates)), 
                                       replace=False)
            for node_id in infected:
                self.nodes[node_id].infected = True
    
    def get_infected_nodes(self) -> Set[int]:
        """Get set of infected node IDs"""
        return {nid for nid, node in self.nodes.items() if node.infected}
    
    def _begin_infection_level(self):
        self.packets_archive.extend(self.packets)
        self.packets.clear()
        self.current_time = 0
        self.drop_stats = Counter()
        self.q_updates.clear()

    def simulate_step(self, infected_nodes: Set[int]):
        """
        Simulation step:
        1. Generate packets
        2. Forward all active packets
        3. Update trust scores
        """
        # Generate packets from valid sources
        new_packets = self.generate_packets(infected_nodes)
        self.packets.extend(new_packets)

        # Forward all active packets
        remaining = []
        for packet in self.packets:
            if not packet.is_delivered and not packet.is_dropped:
                self.forward_packet(packet, infected_nodes)

            if packet.is_delivered or packet.is_dropped:
                self.packets_archive.append(packet)
            else:
                remaining.append(packet)

        self.packets = remaining

        # Periodically normalize trust using accumulated routing statistics.
        for node_id in range(self.params['num_nodes'] + 1):
            node = self.nodes[node_id]
            trust = self.trust_manager.update_trust(
                node_id=node_id,
                packets_sent=node.packets_sent,
                packets_delivered=node.packets_delivered,
                packets_dropped=node.packets_dropped,
                packets_forwarded=node.packets_forwarded,
                current_energy=node.energy,
                initial_energy=node.initial_energy,
                route_changes=node.route_changes,
                is_infected=node.infected
            )
            node.trust_score = trust

        self.current_time += 1
    
    def run_simulation(self):
        """Run complete simulation"""
        print("\nStarting refactored simulation...")
        print("=" * 80)
        
        for level_idx, infection_level in enumerate(self.params['infection_levels']):
            self._begin_infection_level()
            self.introduce_infection(infection_level)
            infected_nodes = self.get_infected_nodes()
            
            print(f"\n[{level_idx + 1}/7] Infection: {infection_level}%")
            
            # Track the starting index of packets for this infection level
            level_start_idx = len(self.packets_archive)
            
            for frame in range(self.params['frames_per_infection_level']):
                self.simulate_step(infected_nodes)
                
                if (frame + 1) % 30 == 0:
                    print(f"  Frame {frame + 1}/{self.params['frames_per_infection_level']}")
            
            self._calculate_metrics(infection_level, infected_nodes, level_start_idx)
            
            pdr = self.metrics.pdr[-1]
            delay = self.metrics.avg_delay[-1]
            connectivity = self.metrics.connectivity[-1]
            print(f"  Results: PDR={pdr:.1f}%, Delay={delay:.1f}ms, Connectivity={connectivity:.1f}%")
        
        print("\n" + "=" * 80)
        print("Simulation complete!")
    
    def _calculate_metrics(self, infection_level: int, infected_nodes: Set[int], level_start_idx: int):
        """Calculate metrics for this infection level"""
        # Only evaluate packets generated during this infection level.
        all_packets = self.packets_archive[level_start_idx:]
        
        delivered = sum(1 for p in all_packets if p.is_delivered)
        total = len(all_packets)
        pdr = 100 * delivered / total if total > 0 else 0
        
        delays = [p.delivered_time - p.created_time for p in all_packets 
                 if p.is_delivered]
        avg_delay = np.mean(delays) if delays else 0
        
        hops = [p.hop_count for p in all_packets if p.is_delivered]
        avg_hops = np.mean(hops) if hops else 0
        
        # Connectivity
        connected = set([0])
        queue = deque([0])
        while queue:
            node_id = queue.popleft()
            for neighbor in self.nodes[node_id].neighbors:
                if neighbor not in connected and neighbor not in infected_nodes:
                    connected.add(neighbor)
                    queue.append(neighbor)
        
        connectivity = 100 * len(connected) / self.params['num_nodes']
        energy = np.mean([n.energy for n in self.nodes.values()])
        
        trust_scores = self.trust_manager.get_all_trust_scores()
        avg_trust = np.mean([t for nid, t in trust_scores.items() if nid != 0])
        
        reward = np.mean([u['reward'] for u in self.q_updates[-100:]]) \
                 if self.q_updates else 0
        
        self.metrics.infection_percentage.append(infection_level)
        self.metrics.pdr.append(pdr)
        self.metrics.avg_delay.append(avg_delay)
        self.metrics.avg_hop_count.append(avg_hops)
        self.metrics.connectivity.append(connectivity)
        self.metrics.residual_energy.append(energy)
        self.metrics.connected_nodes.append(len(connected))
        self.metrics.delivered_packets.append(delivered)
        self.metrics.dropped_packets.append(total - delivered)
        self.metrics.avg_reward.append(reward)
        self.metrics.avg_trust.append(avg_trust)


# ============================================================================
# ANIMATION (UNCHANGED FROM PREVIOUS)
# ============================================================================

class NetworkAnimator:
    """Animation engine - unchanged from Phase 2"""
    
    def __init__(self, network: IoTMeshNetwork, 
                output_file: str = 'outputs/network_animation.mp4'):
        self.network = network
        self.output_file = output_file
        self.fig = None
        self.axes = {}
        self.frames_data = []
        self._prepare_frames()
    
    def _prepare_frames(self):
        """Prepare animation frames"""
        levels = self.network.params['infection_levels']
        frames_per = self.network.params['frames_per_infection_level']
        
        for level_idx, level in enumerate(levels):
            for frame in range(frames_per):
                np.random.seed(level)
                num_inf = int(level / 100.0 * self.network.params['num_nodes'])
                candidates = list(range(1, self.network.params['num_nodes'] + 1))
                infected = set()
                if num_inf > 0:
                    infected = set(np.random.choice(candidates, 
                                                   min(num_inf, len(candidates)), 
                                                   replace=False))
                
                self.frames_data.append({
                    'level': level,
                    'frame': frame,
                    'infected': infected,
                    'metrics_idx': level_idx
                })
    
    def create_animation(self):
        """Create and save animation"""
        self.fig = plt.figure(figsize=(20, 11), dpi=96)
        
        ax_network = plt.subplot(2, 3, (1, 4))
        self.axes['network'] = ax_network
        
        for i, title in enumerate(['pdr', 'delay', 'connectivity', 'energy']):
            ax = plt.subplot(2, 3, 2 + i)
            self.axes[title] = ax
        
        anim = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=len(self.frames_data),
            interval=1000/self.network.params['animation_fps'],
            repeat=False,
            blit=False
        )
        
        print(f"\nGenerating animation ({len(self.frames_data)} frames)...")
        writer = animation.FFMpegWriter(fps=self.network.params['animation_fps'], 
                                       bitrate=5000)
        anim.save(self.output_file, writer=writer, dpi=100)
        print(f"Saved to {self.output_file}")
        plt.close(self.fig)
    
    def _update_frame(self, frame_idx: int):
        """Update single frame"""
        if frame_idx >= len(self.frames_data):
            return
        
        frame = self.frames_data[frame_idx]
        for ax in self.axes.values():
            ax.clear()
        
        self._draw_network(self.axes['network'], frame)
        
        if frame['metrics_idx'] < len(self.network.metrics.pdr):
            self._draw_metrics(frame['metrics_idx'])
        
        plt.tight_layout()
    
    def _draw_network(self, ax, frame):
        """Draw network topology"""
        network = self.network
        nodes = network.nodes
        pos = {n: (nodes[n].x, nodes[n].y) for n in nodes}
        
        # Edges
        for edge in network.graph.edges():
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            ax.plot(x, y, 'gray', alpha=0.2, linewidth=0.5, zorder=1)
        
        # Sink
        sink = nodes[0]
        ax.scatter(sink.x, sink.y, s=400, c='gold', marker='*', 
                  edgecolors='black', linewidth=2, zorder=4)
        
        # Healthy
        healthy = [n for n in nodes if not nodes[n].is_sink and 
                  n not in frame['infected']]
        if healthy:
            hx, hy = zip(*[pos[n] for n in healthy])
            ax.scatter(hx, hy, s=80, c='#00dd88', marker='o', 
                      edgecolors='darkgreen', linewidth=1, zorder=3)
        
        # Infected
        if frame['infected']:
            ix, iy = zip(*[pos[n] for n in frame['infected']])
            ax.scatter(ix, iy, s=80, c='#ff4444', marker='X', 
                      edgecolors='darkred', linewidth=1, zorder=3)
        
        ax.set_xlim(-1, network.params['area_width'] + 1)
        ax.set_ylim(-1, network.params['area_height'] + 1)
        ax.set_xlabel('Distance (km)', fontweight='bold')
        ax.set_ylabel('Distance (km)', fontweight='bold')
        ax.set_title(f'Refactored Geographic Routing | Infection: {frame["level"]}%', 
                    fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
    
    def _draw_metrics(self, idx: int):
        """Draw metric plots"""
        m = self.network.metrics
        
        self.axes['pdr'].plot(m.infection_percentage[:idx+1], m.pdr[:idx+1], 
                             'o-', color='#00dd88', linewidth=2)
        self.axes['pdr'].set_ylabel('PDR %', fontweight='bold')
        self.axes['pdr'].grid(True, alpha=0.3)
        self.axes['pdr'].set_ylim(0, 105)
        
        self.axes['delay'].plot(m.infection_percentage[:idx+1], m.avg_delay[:idx+1], 
                               'o-', color='#00ffff', linewidth=2)
        self.axes['delay'].set_ylabel('Delay (ms)', fontweight='bold')
        self.axes['delay'].grid(True, alpha=0.3)
        
        self.axes['connectivity'].plot(m.infection_percentage[:idx+1], 
                                      m.connectivity[:idx+1], 'o-', 
                                      color='#aa96da', linewidth=2)
        self.axes['connectivity'].set_ylabel('Connectivity %', fontweight='bold')
        self.axes['connectivity'].grid(True, alpha=0.3)
        self.axes['connectivity'].set_ylim(0, 105)
        
        self.axes['energy'].plot(m.infection_percentage[:idx+1], 
                                m.residual_energy[:idx+1], 'o-', 
                                color='#f38181', linewidth=2)
        self.axes['energy'].set_ylabel('Energy (J)', fontweight='bold')
        self.axes['energy'].grid(True, alpha=0.3)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    os.makedirs("outputs", exist_ok=True)
    
    print("\n" + "=" * 80)
    print("REFACTORED: Trust-Aware Geographic Routing with Dynamic Per-Packet Routing")
    print("=" * 80)
    
    print("\n[1/4] Initializing network...")
    network = IoTMeshNetwork(SIMULATION_PARAMS, debug=False)
    
    print("\n[2/4] Running simulation...")
    network.run_simulation()
    
    total = len(network.packets_archive)
    delivered = sum(1 for p in network.packets_archive if p.is_delivered)
    pdr = 100 * delivered / total if total > 0 else 0
    
    print(f"\nTotal packets: {total}")
    print(f"Delivered: {delivered} ({pdr:.1f}%)")
    print(f"\nDrop breakdown:")
    for reason, count in sorted(network.drop_stats.items()):
        print(f"  {reason}: {count}")
    
    print("\n[3/4] Generating animation...")
    animator = NetworkAnimator(network)
    animator.create_animation()
    
    print("\n[4/4] Saving metrics...")
    df = pd.DataFrame({
        'Infection %': network.metrics.infection_percentage,
        'PDR %': network.metrics.pdr,
        'Avg Delay (ms)': network.metrics.avg_delay,
        'Avg Hops': network.metrics.avg_hop_count,
        'Connectivity %': network.metrics.connectivity,
        'Residual Energy': network.metrics.residual_energy,
        'Connected Nodes': network.metrics.connected_nodes,
        'Delivered': network.metrics.delivered_packets,
        'Dropped': network.metrics.dropped_packets,
        'Avg Reward': network.metrics.avg_reward,
        'Avg Trust': network.metrics.avg_trust,
    })
    
    df.to_csv('outputs/simulation_metrics.csv', index=False)
    
    print("\n" + "=" * 80)
    print("REFACTORED RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n✓ Animation: outputs/network_animation.mp4")
    print("✓ Metrics: outputs/simulation_metrics.csv")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()