"""
Trust Management System 
"""

from typing import Dict, Tuple
import numpy as np


# ============================================================================
# TRUST CONFIGURATION CONSTANTS
# ============================================================================

# Trust component weights
PDR_WEIGHT = 0.4
ENERGY_WEIGHT = 0.3
BEHAVIOUR_WEIGHT = 0.3

# Trust bonus in reward function
TRUST_WEIGHT = 10.0  # Reward bonus per trust point

# Trust threshold for threat detection
TRUST_THRESHOLD = 0.4

# Threat penalty when trust drops below threshold
THREAT_PENALTY = 5.0

# Behaviour scoring parameters
MAX_ROUTE_CHANGES_ACCEPTABLE = 5  # Per observation window
MAX_PACKET_DROP_RATE = 0.2  # 20% drop is concerning
INFECTED_PENALTY = 0.3  # Behaviour score reduction

# Trust initialization
INITIAL_TRUST = 1.0


# ============================================================================
# TRUST MANAGER CLASS
# ============================================================================

class TrustManager:
    """
    Lightweight trust manager for IoT mesh networks.
    
    Maintains per-node trust scores based on:
    - Packet Delivery Ratio (PDR)
    - Energy Availability
    - Forwarding Behaviour Consistency
    
    Thread-safe: Updates are independent per node
    Memory-efficient: O(n) storage, O(1) per update
    Publication-ready: Suitable for IEEE/Nature-grade research
    """
    
    def __init__(self, num_nodes: int):
        """
        Initialize trust manager.
        
        Args:
            num_nodes: Total number of nodes in network
        """
        self.num_nodes = num_nodes
        
        # Per-node tracking (indexed by node_id)
        self.pdr_trust: Dict[int, float] = {}
        self.energy_trust: Dict[int, float] = {}
        self.behaviour_trust: Dict[int, float] = {}
        self.overall_trust: Dict[int, float] = {}
        
        # Initialize all nodes
        for node_id in range(num_nodes + 1):
            self.pdr_trust[node_id] = INITIAL_TRUST
            self.energy_trust[node_id] = INITIAL_TRUST
            self.behaviour_trust[node_id] = INITIAL_TRUST
            self.overall_trust[node_id] = INITIAL_TRUST
    
    # ========================================================================
    # PDR TRUST COMPONENT
    # ========================================================================
    
    def update_pdr(self, node_id: int, packets_sent: int, packets_delivered: int):
        """
        Update PDR trust component.
        
        PDR_Trust = packets_delivered / packets_sent
        
        Args:
            node_id: Node identifier
            packets_sent: Total packets sent by this node
            packets_delivered: Packets successfully delivered to sink
        """
        if packets_sent == 0:
            pdr = 1.0  # Neutral if no packets sent yet
        else:
            pdr = min(1.0, packets_delivered / packets_sent)
        
        # Clamp to [0, 1]
        self.pdr_trust[node_id] = np.clip(pdr, 0.0, 1.0)
    
    # ========================================================================
    # ENERGY TRUST COMPONENT
    # ========================================================================
    
    def update_energy(self, node_id: int, current_energy: float, initial_energy: float):
        """
        Update energy trust component.
        
        Energy_Trust = current_energy / initial_energy
        
        Args:
            node_id: Node identifier
            current_energy: Remaining energy in joules
            initial_energy: Initial energy capacity in joules
        """
        if initial_energy <= 0:
            energy_trust = 1.0
        else:
            energy_trust = current_energy / initial_energy
        
        # Clamp to [0, 1]
        self.energy_trust[node_id] = np.clip(energy_trust, 0.0, 1.0)
    
    # ========================================================================
    # BEHAVIOUR TRUST COMPONENT
    # ========================================================================
    
    def update_behaviour(self, node_id: int, route_changes: int,
                        packets_dropped: int, packets_forwarded: int,
                        is_infected: bool):
        """
        Update behaviour trust component.
        
        Derived from:
        - Route change frequency (stability)
        - Packet drop rate (reliability)
        - Infected status (compromise risk)
        - Forwarding consistency
        
        Args:
            node_id: Node identifier
            route_changes: Number of route changes observed
            packets_dropped: Total packets dropped by this node
            packets_forwarded: Total packets forwarded by this node
            is_infected: Whether node is currently compromised
        """
        behaviour_score = 1.0
        
        # FACTOR 1: Route stability
        # Excessive route changes reduce trust
        if route_changes > MAX_ROUTE_CHANGES_ACCEPTABLE:
            excess = route_changes - MAX_ROUTE_CHANGES_ACCEPTABLE
            stability_penalty = min(0.3, 0.05 * excess)  # Max 0.3 penalty
            behaviour_score -= stability_penalty
        
        # FACTOR 2: Drop rate
        # High packet drop rate reduces trust
        if packets_forwarded > 0:
            drop_rate = packets_dropped / packets_forwarded
            if drop_rate > MAX_PACKET_DROP_RATE:
                excess_rate = drop_rate - MAX_PACKET_DROP_RATE
                drop_penalty = min(0.4, excess_rate * 2)  # Max 0.4 penalty
                behaviour_score -= drop_penalty
        
        # FACTOR 3: Infection status
        # Compromised nodes are untrustworthy
        if is_infected:
            behaviour_score -= INFECTED_PENALTY
        
        # Clamp to [0, 1]
        self.behaviour_trust[node_id] = np.clip(behaviour_score, 0.0, 1.0)
    
    # ========================================================================
    # OVERALL TRUST CALCULATION
    # ========================================================================
    
    def calculate_trust(self, node_id: int) -> float:
        """
        Calculate overall trust score using weighted combination.
        
        Trust = 0.4 × PDR_Trust + 0.3 × Energy_Trust + 0.3 × Behaviour_Trust
        
        Args:
            node_id: Node identifier
            
        Returns:
            Overall trust score in [0, 1]
        """
        pdr = self.pdr_trust.get(node_id, INITIAL_TRUST)
        energy = self.energy_trust.get(node_id, INITIAL_TRUST)
        behaviour = self.behaviour_trust.get(node_id, INITIAL_TRUST)
        
        overall = (PDR_WEIGHT * pdr + 
                  ENERGY_WEIGHT * energy + 
                  BEHAVIOUR_WEIGHT * behaviour)
        
        # Clamp final trust to [0, 1]
        return np.clip(overall, 0.0, 1.0)
    
    # ========================================================================
    # UPDATE INTERFACE
    # ========================================================================
    
    def update_trust(self, node_id: int, packets_sent: int, packets_delivered: int,
                    packets_dropped: int, packets_forwarded: int,
                    current_energy: float, initial_energy: float,
                    route_changes: int, is_infected: bool) -> float:
        """
        Unified update method - updates all trust components and returns overall trust.
        
        This is the primary interface for integrating trust into simulation.
        Call this once per simulation step for each node.
        
        Args:
            node_id: Node identifier
            packets_sent: Packets sent by this node
            packets_delivered: Packets delivered to sink
            packets_dropped: Packets dropped by this node
            packets_forwarded: Packets forwarded through this node
            current_energy: Current node energy (J)
            initial_energy: Initial node energy (J)
            route_changes: Number of route changes
            is_infected: Infection status
            
        Returns:
            Updated overall trust score in [0, 1]
        """
        # Update component scores
        self.update_pdr(node_id, packets_sent, packets_delivered)
        self.update_energy(node_id, current_energy, initial_energy)
        self.update_behaviour(node_id, route_changes, packets_dropped,
                            packets_forwarded, is_infected)
        
        # Calculate and store overall trust
        trust = self.calculate_trust(node_id)
        self.overall_trust[node_id] = trust
        
        return trust
    
    # ========================================================================
    # QUERY INTERFACE
    # ========================================================================
    
    def get_trust(self, node_id: int) -> float:
        """
        Get current overall trust score for a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Trust score in [0, 1]
        """
        return self.overall_trust.get(node_id, INITIAL_TRUST)
    
    def get_all_trust_scores(self) -> Dict[int, float]:
        """
        Get all current trust scores.
        
        Returns:
            Dictionary mapping node_id → trust_score
        """
        return self.overall_trust.copy()
    
    def get_trust_components(self, node_id: int) -> Dict[str, float]:
        """
        Get detailed trust component breakdown for a node.
        
        Useful for analysis and debugging.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Dictionary with 'pdr', 'energy', 'behaviour', 'overall' scores
        """
        return {
            'pdr': self.pdr_trust.get(node_id, INITIAL_TRUST),
            'energy': self.energy_trust.get(node_id, INITIAL_TRUST),
            'behaviour': self.behaviour_trust.get(node_id, INITIAL_TRUST),
            'overall': self.overall_trust.get(node_id, INITIAL_TRUST),
        }
    
    def identify_low_trust_nodes(self, threshold: float = 0.5) -> list:
        """
        Identify nodes with trust below threshold.
        
        Useful for security monitoring and analysis.
        
        Args:
            threshold: Trust threshold
            
        Returns:
            List of (node_id, trust_score) tuples for low-trust nodes
        """
        low_trust = [(nid, trust) for nid, trust in self.overall_trust.items()
                     if trust < threshold]
        return sorted(low_trust, key=lambda x: x[1])
    
    # ========================================================================
    # STATISTICAL SUMMARIES
    # ========================================================================
    
    def get_network_trust_statistics(self) -> Dict[str, float]:
        """
        Get network-wide trust statistics.
        
        Returns:
            Dictionary with 'mean', 'std', 'min', 'max' trust scores
        """
        trust_values = list(self.overall_trust.values())
        if not trust_values:
            return {'mean': 1.0, 'std': 0.0, 'min': 1.0, 'max': 1.0}
        
        trust_array = np.array(trust_values)
        return {
            'mean': float(np.mean(trust_array)),
            'std': float(np.std(trust_array)),
            'min': float(np.min(trust_array)),
            'max': float(np.max(trust_array)),
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_trust_bonus(trust_score: float, weight: float = TRUST_WEIGHT) -> float:
    """
    Compute reward bonus based on trust score.
    
    Args:
        trust_score: Node trust score in [0, 1]
        weight: Bonus weight parameter
        
    Returns:
        Bonus value to add to reward
    """
    return weight * trust_score


def compute_threat_penalty(trust_score: float, threshold: float = TRUST_THRESHOLD,
                          penalty: float = THREAT_PENALTY) -> float:
    """
    Compute penalty for low-trust nodes.
    
    Args:
        trust_score: Node trust score in [0, 1]
        threshold: Trust threshold
        penalty: Penalty magnitude
        
    Returns:
        Penalty value to subtract from reward
    """
    if trust_score < threshold:
        return penalty
    return 0.0


# ============================================================================
# MATHEMATICAL FORMULATION (for publication)
# ============================================================================

"""
TRUST MODEL - Mathematical Formulation

1. PDR TRUST COMPONENT
   ────────────────────
   τ_pdr(i) = d_i / s_i
   
   where:
     τ_pdr(i) ∈ [0,1] is PDR trust for node i
     d_i = delivered packets by node i
     s_i = sent packets by node i
   
   Interpretation: Higher delivery ratio → higher trust
   Domain: [0, 1] normalized

2. ENERGY TRUST COMPONENT
   ──────────────────────
   τ_energy(i) = e_i(t) / e_i(0)
   
   where:
     τ_energy(i) ∈ [0,1] is energy trust for node i
     e_i(t) = current energy at time t
     e_i(0) = initial energy
   
   Interpretation: Higher remaining energy → higher trust
   Domain: [0, 1] normalized

3. BEHAVIOUR TRUST COMPONENT
   ─────────────────────────
   τ_behaviour(i) = 1 - α·Δr(i) - β·d_rate(i) - γ·I(i)
   
   where:
     Δr(i) = normalized route change frequency
     d_rate(i) = packet drop rate
     I(i) = infection indicator (0 if healthy, 1 if infected)
     α, β, γ = penalty weights
   
   Interpretation: Stable, reliable, uncompromised nodes have higher trust
   Domain: [0, 1] normalized

4. OVERALL TRUST SCORE
   ────────────────────
   τ(i,t) = w₁·τ_pdr(i,t) + w₂·τ_energy(i,t) + w₃·τ_behaviour(i,t)
   
   where:
     w₁ = 0.4 (PDR weight)
     w₂ = 0.3 (Energy weight)
     w₃ = 0.3 (Behaviour weight)
     w₁ + w₂ + w₃ = 1.0
   
   Domain: τ(i,t) ∈ [0, 1]
   Constraints: τ(i,t) is clipped to [0, 1]

5. TRUST-AUGMENTED REWARD FUNCTION
   ────────────────────────────────
   R_total(i→j) = R_base(i→j) + ω·τ(j,t) - ρ·𝟙[τ(j,t) < θ]
   
   where:
     R_base(i→j) = existing routing reward
     ω = 10.0 (trust bonus weight)
     τ(j,t) = trust of next-hop node j
     ρ = 5.0 (threat penalty)
     θ = 0.4 (trust threshold)
     𝟙[·] = indicator function
   
   Interpretation:
     - Positive reward bonus proportional to next-hop trust
     - Additional penalty when next-hop trust is critically low
     - Incentivizes routing through trusted nodes
"""
