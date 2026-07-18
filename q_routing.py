"""
Q-Learning Routing Implementation for IoT Mesh Networks
"""

import math
import random
import csv
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from network import IoTNetwork, Node, TopologyType


class QRoutingAgent:
    """
    Q-Learning routing agent for IoT networks.

    Parameters from paper:
    - Learning rate (alpha): 0.7
    - Training episodes: 900
    - Initial Q-value: 100
    - Discount factor (gamma): neighbor-set based (Eq 7)
    """

    def __init__(
        self,
        network: IoTNetwork,
        alpha: float = 0.7,
        epsilon: float = 0.1,
        gamma_max: float = 100.0,
        gamma_min: float = -100.0,
        initial_q_value: float = 100.0,
    ):
        """
        Initialize Q-routing agent.

        Args:
            network: IoTNetwork instance
            alpha: Learning rate (paper specifies 0.7)
            epsilon: Exploration probability (not in paper, must be configurable)
            gamma_max: Maximum reward (destination reached)
            gamma_min: Minimum reward (local minima)
            initial_q_value: Initial Q-table value (paper specifies 100)
        """
        self.network = network
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        self.initial_q_value = initial_q_value

        # Q-table: Q[state][action] where state=(current_node, dest), action=next_hop
        # FIX P4: state is (current_node, destination), NOT (source, current_node)
        self.q_table: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(
            lambda: defaultdict(lambda: initial_q_value)
        )

        # Metrics tracking
        self.training_log: List[Dict] = []
        self.episode_rewards: List[float] = []

        # Reward function parameters (Eq 2)
        self.omega = 0.5  # Not specified in paper, configurable weight

        # Initialize Q-table for all valid state-action pairs
        self._initialize_q_table()

    def _initialize_q_table(self) -> None:
        """
        Initialize Q-table with all valid state-action pairs.

        State: (current_node, destination_node)   ← per paper Eq (6)
        Action: next_hop_neighbor
        Initial Q-value: 100 (as per paper)
        """
        nodes = list(self.network.nodes.keys()) + [-1]  # Include sink

        for current in nodes:
            for dest in nodes:
                if current != dest:
                    current_node = self.network.get_node(current)
                    if current_node:
                        for neighbor in current_node.neighbor_list:
                            self.q_table[(current, dest)][neighbor] = self.initial_q_value

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def euclidean_distance(node1: Node, node2: Node) -> float:
        """Calculate Euclidean distance between nodes."""
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def evaluate_angle(
        self,
        current_node: Node,
        neighbor_node: Node,
        sink_node: Node,
    ) -> float:
        """
        Evaluate angle between reference line (current→sink) and neighbor direction.

        Equation (11): Uses vector dot product for minimum angle routing.

        Angle = arccos(dot_product / (||vector1|| * ||vector2||))

        Args:
            current_node: Current node in path
            neighbor_node: Candidate neighbor
            sink_node: Destination (sink)

        Returns:
            Angle in radians
        """
        ref_x = sink_node.x - current_node.x
        ref_y = sink_node.y - current_node.y
        ref_length = math.sqrt(ref_x ** 2 + ref_y ** 2)

        neighbor_x = neighbor_node.x - current_node.x
        neighbor_y = neighbor_node.y - current_node.y
        neighbor_length = math.sqrt(neighbor_x ** 2 + neighbor_y ** 2)

        if ref_length == 0 or neighbor_length == 0:
            return math.pi

        dot_product = ref_x * neighbor_x + ref_y * neighbor_y
        cos_angle = dot_product / (ref_length * neighbor_length)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.acos(cos_angle)

    def minimum_angle_node(
        self,
        current_node_id: int,
        destination_node_id: int,
        path: List[int],
    ) -> Optional[Tuple[int, float]]:
        """
        Select next hop using minimum angle routing.

        Returns neighbor with smallest angle to sink (Equation 11).
        Avoids nodes already in path (prevent loops).

        Args:
            current_node_id: Current node
            destination_node_id: Destination (sink)
            path: Current path (to prevent loops)

        Returns:
            Tuple of (neighbor_id, angle) or None if no valid neighbor
        """
        current_node = self.network.get_node(current_node_id)
        destination_node = self.network.get_node(destination_node_id)

        if not current_node or not destination_node:
            return None

        valid_neighbors = [n for n in current_node.neighbor_list if n not in path]
        if not valid_neighbors:
            return None

        min_angle = float('inf')
        best_neighbor = None

        for neighbor_id in valid_neighbors:
            neighbor_node = self.network.get_node(neighbor_id)
            if neighbor_node:
                angle = self.evaluate_angle(current_node, neighbor_node, destination_node)
                if angle < min_angle:
                    min_angle = angle
                    best_neighbor = neighbor_id

        if best_neighbor is not None:
            return (best_neighbor, min_angle)
        return None

    def get_sorted_neighbors_by_angle(
        self,
        current_node_id: int,
        destination_node_id: int,
        path: List[int],
    ) -> List[Tuple[int, float]]:
        """
        Get all valid neighbors sorted by angle (ascending).

        Used for void handling (Algorithm 5).

        Args:
            current_node_id: Current node
            destination_node_id: Destination
            path: Current path

        Returns:
            List of (neighbor_id, angle) tuples sorted by angle
        """
        current_node = self.network.get_node(current_node_id)
        destination_node = self.network.get_node(destination_node_id)

        if not current_node or not destination_node:
            return []

        valid_neighbors = [n for n in current_node.neighbor_list if n not in path]
        angles = []
        for neighbor_id in valid_neighbors:
            neighbor_node = self.network.get_node(neighbor_id)
            if neighbor_node:
                angle = self.evaluate_angle(current_node, neighbor_node, destination_node)
                angles.append((neighbor_id, angle))

        return sorted(angles, key=lambda x: x[1])

    # ------------------------------------------------------------------
    # Void handling — Algorithm 5
    # ------------------------------------------------------------------

    def void_handling(
        self,
        current_node_id: int,
        destination_node_id: int,
        path: List[int],
    ) -> Optional[int]:
        """
        Implement void handling (Algorithm 5 from paper).

        Called when ALL valid (unvisited) neighbors are exhausted.

        Strategy (path-aware version):
          Step 1 — min-angle neighbor NOT in path (unvisited preferred)
          Step 2 — 2nd-angle neighbor NOT in path     [P1 FIX]
          Step 3 — 3rd-angle neighbor NOT in path     [P1 FIX]
          Step 2b — 2nd-angle neighbor (allow revisit as last resort)
          Step 3b — 3rd-angle neighbor (allow revisit as last resort)
          Step 4 — minimum-distance-to-sink neighbor (any)
          Step 5 — return None (truly stuck)

        P1 FIX: Steps 2 and 3 now first try the unvisited candidate at that
        angle rank before falling back to the visited one. This avoids
        unnecessarily returning a visited node when a better unvisited
        option exists at a slightly larger angle.

        Args:
            current_node_id: Current node
            destination_node_id: Destination
            path: Current path

        Returns:
            Next hop node ID or None
        """
        current_node = self.network.get_node(current_node_id)
        if not current_node:
            return None

        destination_node = self.network.get_node(destination_node_id)
        if not destination_node:
            return None

        # Build angle-sorted list of ALL neighbors (no path filter yet)
        all_neighbors_sorted: List[Tuple[int, float]] = []
        for neighbor_id in current_node.neighbor_list:
            neighbor_node = self.network.get_node(neighbor_id)
            if neighbor_node and neighbor_id != current_node_id:
                angle = self.evaluate_angle(current_node, neighbor_node, destination_node)
                all_neighbors_sorted.append((neighbor_id, angle))
        all_neighbors_sorted.sort(key=lambda x: x[1])

        # Step 1: minimum-angle unvisited neighbor
        for neighbor_id, _ in all_neighbors_sorted:
            if neighbor_id not in path:
                return neighbor_id

        # ----------------------------------------------------------------
        # All neighbors are in path.  Fall back to angle-ranked revisits.
        # P1 FIX: prefer unvisited at each rank before allowing revisit.
        # ----------------------------------------------------------------

        # Step 2: 2nd-angle neighbor — prefer unvisited, else allow revisit
        if len(all_neighbors_sorted) >= 2:
            second_id = all_neighbors_sorted[1][0]
            if second_id != current_node_id and second_id not in path:
                return second_id  # unvisited at rank 2

        # Step 3: 3rd-angle neighbor — prefer unvisited, else allow revisit
        if len(all_neighbors_sorted) >= 3:
            third_id = all_neighbors_sorted[2][0]
            if third_id != current_node_id and third_id not in path:
                return third_id  # unvisited at rank 3

        # Step 2b: no unvisited found at ranks 2/3 — allow revisit at rank 2
        if len(all_neighbors_sorted) >= 2:
            second_id = all_neighbors_sorted[1][0]
            if second_id != current_node_id:
                return second_id

        # Step 3b: allow revisit at rank 3
        if len(all_neighbors_sorted) >= 3:
            third_id = all_neighbors_sorted[2][0]
            if third_id != current_node_id:
                return third_id

        # Step 4: minimum-distance-to-sink fallback (Algorithm 5, paper)
        min_dist = float('inf')
        closest_neighbor = None
        for neighbor_id in current_node.neighbor_list:
            if neighbor_id != current_node_id:
                neighbor_node = self.network.get_node(neighbor_id)
                if neighbor_node:
                    dist = self.euclidean_distance(neighbor_node, destination_node)
                    if dist < min_dist:
                        min_dist = dist
                        closest_neighbor = neighbor_id

        if closest_neighbor is not None:
            return closest_neighbor

        # Step 5: truly stuck
        return None

    # ------------------------------------------------------------------
    # Link quality metrics — Equations 12–14
    # ------------------------------------------------------------------

    def compute_link_quality(
        self,
        from_node_id: int,
        to_node_id: int,
        forward_delivery_ratio: float = 0.95,
        reverse_delivery_ratio: float = 0.95,
    ) -> float:
        """
        Compute link quality (Equation 13).

        QL = Fd * Rd
        """
        return forward_delivery_ratio * reverse_delivery_ratio

    def neighbor_coefficient(
        self,
        from_node_id: int,
        to_node_id: int,
    ) -> float:
        """
        Compute neighbor relationship coefficient (Equation 14).

        N(i,j) = 1 - (distance / transmission_range)  if distance <= range
                 else 0
        """
        from_node = self.network.get_node(from_node_id)
        to_node = self.network.get_node(to_node_id)

        if not from_node or not to_node:
            return 0.0

        distance = self.euclidean_distance(from_node, to_node)
        if distance > self.network.transmission_range:
            return 0.0

        return 1.0 - (distance / self.network.transmission_range)

    def weighted_q_value(
        self,
        from_node_id: int,
        to_node_id: int,
        q_value: float,
    ) -> float:
        """
        Compute weighted Q-value for routing decision (Equation 12).

        WT = NeighborCoefficient * LinkQuality * Q_value
        """
        nc = self.neighbor_coefficient(from_node_id, to_node_id)
        lq = self.compute_link_quality(from_node_id, to_node_id)
        return nc * lq * q_value

    # ------------------------------------------------------------------
    # Reward and discount — Equations 2 and 7
    # ------------------------------------------------------------------

    def reward_function(
        self,
        current_node_id: int,
        next_node_id: int,
        destination_node_id: int,
        delay: float,
        local_minima: bool = False,
    ) -> float:
        """
        Compute reward for reaching next node (Equation 2).

        Reward =
            gamma_max         if destination reached
            gamma_min         if local minima
            omega * exp(-delay) + (1-omega) * residual_energy_ratio
                              otherwise

        Args:
            delay: Per-link delay for this hop (not path average)  ← P4 fix
        """
        if next_node_id == destination_node_id:
            return self.gamma_max

        if local_minima:
            return self.gamma_min

        next_node = self.network.get_node(next_node_id)
        if not next_node:
            return self.gamma_min

        residual_energy_ratio = next_node.energy_ratio()
        delay_component = self.omega * math.exp(-delay)
        energy_component = (1 - self.omega) * residual_energy_ratio

        return delay_component + energy_component

    def compute_discount_factor(
        self,
        current_node_id: int,
        next_node_id: int,
    ) -> float:
        """
        Compute neighbor-set based discount factor (Equation 7).

        gamma_i = 1 - (|Union| - |Intersection|) / |Union|
        """
        current_node = self.network.get_node(current_node_id)
        next_node = self.network.get_node(next_node_id)

        if not current_node or not next_node:
            return 0.5

        union = current_node.neighbor_list | next_node.neighbor_list
        intersection = current_node.neighbor_list & next_node.neighbor_list

        if len(union) == 0:
            return 0.5

        gamma = 1.0 - (len(union) - len(intersection)) / len(union)
        return max(0.0, min(1.0, gamma))

    # ------------------------------------------------------------------
    # Next-hop selection
    # ------------------------------------------------------------------

    def select_next_hop(
        self,
        current_node_id: int,
        destination_node_id: int,
        path: List[int],
        use_q_learning: bool = True,
    ) -> Optional[int]:
        """
        Select next hop using Q-learning (exploit) or minimum angle routing (explore).

        Returns None only when truly stuck (void handling exhausted).

        P2 FIX: backtrack unexplored-check now uses full `path` (not path[:-1])
        so current_node_id is correctly excluded from "unexplored" candidates.

        Args:
            current_node_id: Current node
            destination_node_id: Destination
            path: Current path
            use_q_learning: Whether to use learned policy

        Returns:
            Next hop node ID or None
        """
        current_node = self.network.get_node(current_node_id)
        if not current_node:
            return None

        valid_neighbors = [n for n in current_node.neighbor_list if n not in path]

        if not valid_neighbors:
            # No unvisited neighbors — try void handling first
            next_hop = self.void_handling(current_node_id, destination_node_id, path)

            if next_hop is not None and next_hop not in path:
                return next_hop

            # Void handling could not find an unvisited node.
            # Try one controlled backtrack: return to previous node so the
            # caller (route_packet) can trim the path and try its other branches.
            # P2 FIX: use full `path` so current_node_id is excluded.
            if len(path) >= 2:
                previous_node = path[-2]
                prev_node_obj = self.network.get_node(previous_node)
                if prev_node_obj:
                    unexplored = [
                        n for n in prev_node_obj.neighbor_list
                        if n not in path          # P2 FIX: was path[:-1]
                    ]
                    if unexplored:
                        return previous_node  # Signal: backtrack to here

            return None

        # We have valid (unvisited) neighbors — run ε-greedy policy
        if use_q_learning and random.random() > self.epsilon:
            # EXPLOIT: maximum weighted Q-value among valid neighbors
            best_neighbor = None
            best_weighted_q = -float('inf')

            for neighbor_id in valid_neighbors:
                q_val = self.q_table[(current_node_id, destination_node_id)].get(
                    neighbor_id, self.initial_q_value
                )
                wq = self.weighted_q_value(current_node_id, neighbor_id, q_val)
                if wq > best_weighted_q:
                    best_weighted_q = wq
                    best_neighbor = neighbor_id

            if best_neighbor is not None and best_neighbor not in path:
                return best_neighbor

        # EXPLORE: minimum-angle routing
        result = self.minimum_angle_node(current_node_id, destination_node_id, path)
        if result:
            neighbor_id = result[0]
            if neighbor_id not in path:
                return neighbor_id

        # FALLBACK: first valid unvisited neighbor
        if valid_neighbors:
            return valid_neighbors[0]

        return None

    # ------------------------------------------------------------------
    # Packet routing
    # ------------------------------------------------------------------

    def route_packet(
        self,
        source_id: int,
        destination_id: int,
        max_hops: int = 50,
    ) -> Tuple[List[int], float]:
        """
        Route a packet from source to destination.

        P3 FIX: Replace visited_count with path-membership cycle detection.
        Allows exactly one controlled backtrack (path trim) without false
        cycle termination. A backtrack trims the path back to the revisited
        node and continues forward from there — it does NOT append a duplicate.

        Returns (path, total_delay).  Path ends with destination_id on success,
        or at the last reached node on failure (explicit dead-end, never silent).

        Args:
            source_id: Source node
            destination_id: Destination (sink, -1)
            max_hops: Maximum hop budget

        Returns:
            Tuple of (path, total_delay)
        """
        path = [source_id]
        total_delay = 0.0
        current_id = source_id

        TRANSMISSION_ENERGY = 0.5  # J (802.15.4 typical)
        RECEPTION_ENERGY = 0.3     # J

        source_node = self.network.get_node(source_id)
        if source_node:
            source_node.packet_sent += 1
            source_node.update_energy(TRANSMISSION_ENERGY)

        # P3 FIX: one backtrack token per route
        backtrack_used = False

        while current_id != destination_id and len(path) < max_hops:
            next_id = self.select_next_hop(
                current_id, destination_id, path, use_q_learning=True
            )

            if next_id is None:
                # Genuinely stuck — explicit failure (no silent termination)
                break

            # ----------------------------------------------------------------
            # P3 FIX: cycle detection via path membership.
            # If next_id is already in path, this is a backtrack signal.
            # Allow it exactly once by trimming the path back to that node.
            # ----------------------------------------------------------------
            if next_id in path:
                if backtrack_used:
                    # Second backtrack attempt → terminate to prevent cycling
                    break
                backtrack_used = True
                # Trim path back to the backtrack node (inclusive)
                backtrack_index = path.index(next_id)
                path = path[:backtrack_index + 1]
                current_id = next_id
                # Do NOT append next_id again; it's already at path[-1]
                continue

            # Normal forward step — compute link metrics
            current_node = self.network.get_node(current_id)
            next_node = self.network.get_node(next_id)

            if current_node and next_node:
                distance = self.euclidean_distance(current_node, next_node)
                delay = distance / 10.0
                total_delay += delay

                current_node.packet_sent += 1
                next_node.packet_received += 1

                current_node.update_energy(TRANSMISSION_ENERGY)
                next_node.update_energy(RECEPTION_ENERGY)

            path.append(next_id)
            current_id = next_id

        # Track sink reception
        if current_id == destination_id and destination_id == -1:
            sink = self.network.get_node(destination_id)
            if sink:
                sink.packet_received += 1

        return path, total_delay

    # ------------------------------------------------------------------
    # Bellman update — Equation 6
    # ------------------------------------------------------------------

    def bellman_update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
    ) -> None:
        """
        Perform Q-table update using Bellman equation (Equation 6).

        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',*)) - Q(s,a))

        where state = (current_node, destination)  ← per paper Eq (6)
              alpha = 0.7

        Args:
            state: (current_node, destination)
            action: next_hop
            reward: Immediate reward
            next_state: (next_node, destination)
        """
        old_q = self.q_table[state].get(action, self.initial_q_value)

        next_q_values = self.q_table[next_state]
        max_next_q = (
            max(next_q_values.values()) if next_q_values else self.initial_q_value
        )

        new_q = old_q + self.alpha * (
            reward
            + self.compute_discount_factor(state[0], action) * max_next_q
            - old_q
        )
        self.q_table[state][action] = new_q

    # ------------------------------------------------------------------
    # Training loop — 900 episodes per paper
    # ------------------------------------------------------------------

    def train(self, episodes: int = 900, max_hops: int = 50) -> None:
        """
        Train Q-routing for specified number of episodes.

        Paper specifies: 900 episodes.

        P4 FIX (state space):
          state = (current_id, destination_id)   ← paper Eq (6)
          NOT  (source_id, current_id)
          The old formulation tied Q-values to the packet's origin, preventing
          generalisation: a relay node learned N_sources separate policies
          instead of one. With 900 episodes and N nodes, the correct formulation
          visits each (node, dest) state O(900) times; the wrong one visits
          each (src, node, dest) triple O(900/N) times.

        P4 FIX (per-link delay):
          reward_function receives the individual link delay for hop i→i+1,
          not total_delay/len(path). The paper's Eq (2) exp(-delay) is a
          per-transition quantity.

        Args:
            episodes: Number of training episodes (default 900 per paper)
            max_hops: Maximum hops per route
        """
        node_ids = list(self.network.nodes.keys())

        for episode in range(episodes):
            source_id = random.choice(node_ids)
            destination_id = -1  # Sink

            path, total_delay = self.route_packet(source_id, destination_id, max_hops)

            # P4 FIX: pre-compute per-link delays for accurate reward signals
            link_delays: List[float] = []
            for i in range(len(path) - 1):
                n1 = self.network.get_node(path[i])
                n2 = self.network.get_node(path[i + 1])
                if n1 and n2:
                    link_delays.append(self.euclidean_distance(n1, n2) / 10.0)
                else:
                    link_delays.append(0.0)

            episode_reward = 0.0

            for i in range(len(path) - 1):
                current_id = path[i]
                next_id = path[i + 1]

                # P4 FIX: per-link delay (not path average)
                link_delay = link_delays[i] if i < len(link_delays) else 0.0

                local_minima = (next_id == current_id)  # Should never occur
                reward = self.reward_function(
                    current_id, next_id, destination_id, link_delay, local_minima
                )
                episode_reward += reward

                # P4 FIX: state = (current_node, destination) per Eq (6)
                state = (current_id, destination_id)
                next_state = (next_id, destination_id)
                self.bellman_update(state, next_id, reward, next_state)

            self.episode_rewards.append(episode_reward)

            if episode % 100 == 0:
                self.training_log.append({
                    'episode': episode,
                    'path_length': len(path),
                    'total_delay': total_delay,
                    'episode_reward': episode_reward,
                })

    # ------------------------------------------------------------------
    # Export and statistics
    # ------------------------------------------------------------------

    def export_q_table(self, filepath: str = "outputs/q_table.csv") -> None:
        """
        Export Q-table to CSV format.

        Format: current_node,destination,next_hop,q_value
        """
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['current_node', 'destination', 'next_hop', 'q_value'])

            for (current_node, dest), actions in self.q_table.items():
                for next_hop, q_value in actions.items():
                    writer.writerow([current_node, dest, next_hop, f"{q_value:.6f}"])

    def export_training_log(self, filepath: str = "outputs/training_log.csv") -> None:
        """Export training metrics to CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'path_length', 'total_delay', 'episode_reward'])
            for log in self.training_log:
                writer.writerow([
                    log['episode'], log['path_length'],
                    log['total_delay'], log['episode_reward'],
                ])

    def compute_routing_statistics(self) -> dict:
        """
        Compute comprehensive routing statistics.

        Returns:
            Dictionary with routing metrics including energy, PDR, route length.
        """
        stats = {
            'successful_routes': 0,
            'failed_routes': 0,
            'sink_reach_rate': 0.0,
            'packet_delivery_ratio': 0.0,
            'avg_residual_energy': 0.0,
            'min_residual_energy': 0.0,
            'max_residual_energy': 0.0,
            'total_energy_consumed': 0.0,
            'avg_route_length': 0.0,
        }

        residual_energies = [node.residual_energy for node in self.network.nodes.values()]
        if residual_energies:
            stats['avg_residual_energy'] = sum(residual_energies) / len(residual_energies)
            stats['min_residual_energy'] = min(residual_energies)
            stats['max_residual_energy'] = max(residual_energies)

        initial_total = sum(node.initial_energy for node in self.network.nodes.values())
        current_total = sum(node.residual_energy for node in self.network.nodes.values())
        stats['total_energy_consumed'] = initial_total - current_total

        total_sent = sum(node.packet_sent for node in self.network.nodes.values())
        total_received = sum(node.packet_received for node in self.network.nodes.values())
        if total_sent > 0:
            stats['packet_delivery_ratio'] = total_received / total_sent

        return stats

    def validate_routes(self, num_routes: int = 10) -> dict:
        """
        Validate routing quality and report statistics.

        A route is classified as:
          - sink_reached:          path ends at destination_id (-1)
          - premature_termination: path ended before sink (explicit failure)
          - routes_with_cycles:    path contains duplicate node IDs
          - valid_routes:          no issues found

        Args:
            num_routes: Number of routes to test

        Returns:
            Dictionary with validation results
        """
        results = {
            'total_routes': num_routes,
            'sink_reached': 0,
            'premature_termination': 0,
            'routes_with_cycles': 0,
            'routes_with_invalid_transitions': 0,
            'valid_routes': 0,
            'route_details': [],
        }

        for source_id in range(min(num_routes, len(self.network.nodes))):
            path, _ = self.route_packet(source_id, -1, max_hops=50)

            route_info = {
                'source': source_id,
                'length': len(path),
                'destination_reached': path[-1] == -1,
                'issues': [],
            }

            if len(path) != len(set(path)):
                route_info['issues'].append('cycle')
                results['routes_with_cycles'] += 1

            for i in range(len(path) - 1):
                curr = path[i]
                nxt = path[i + 1]
                curr_node = self.network.get_node(curr)
                if curr_node and nxt not in curr_node.neighbor_list and nxt != -1:
                    route_info['issues'].append('invalid_transition')
                    results['routes_with_invalid_transitions'] += 1
                    break

            if path[-1] == -1:
                results['sink_reached'] += 1
            else:
                results['premature_termination'] += 1

            if not route_info['issues']:
                results['valid_routes'] += 1

            results['route_details'].append(route_info)

        return results