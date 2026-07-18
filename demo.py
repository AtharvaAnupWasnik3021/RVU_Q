#!/usr/bin/env python3
"""
Complete demonstration of IoT Q-routing implementation.

Reproduces workflow from Chakraborty, Das & Pradhan (2022).
"""

import os
import sys
from network import IoTNetwork, TopologyType
from q_routing import QRoutingAgent
from visualize import NetworkVisualizer, visualize_all_topologies


def main():
    """Run complete demonstration."""
    
    print("\n" + "="*70)
    print("IoT Q-Learning Routing Protocol - Complete Demonstration")
    print("="*70)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # ========================================================================
    # PART 1: Generate and Visualize Topologies
    # ========================================================================
    print("\n[STEP 1] Generating Network Topologies")
    print("-" * 70)
    visualize_all_topologies(output_dir="outputs")
    
    # ========================================================================
    # PART 2: Demonstrate Routing on Mesh Topology
    # ========================================================================
    print("\n[STEP 2] Training Q-Routing Agent (Mesh Topology)")
    print("-" * 70)
    
    network = IoTNetwork(TopologyType.MESH)
    print(f"✓ Created Mesh topology: {len(network.nodes)} nodes, 25×25 km, 4.5 km range")
    print(f"  Sink at: ({network.sink.x}, {network.sink.y})")
    
    # Print network statistics
    stats = network.get_statistics()
    print(f"  Average neighbors per node: {sum(len(n.neighbor_list) for n in network.nodes.values()) / len(network.nodes):.1f}")
    
    # Initialize agent with paper parameters
    agent = QRoutingAgent(
        network,
        alpha=0.7,              # Paper specifies
        epsilon=0.1,            # Not in paper, but sensible default
        gamma_max=100.0,        # Paper specifies
        gamma_min=-100.0,       # Paper specifies
        initial_q_value=100.0,  # Paper specifies
    )
    print(f"\n✓ Initialized Q-routing agent:")
    print(f"  Learning rate (α): {agent.alpha}")
    print(f"  Exploration (ε): {agent.epsilon}")
    print(f"  Initial Q-value: {agent.initial_q_value}")
    print(f"  State-action pairs: {sum(len(a) for a in agent.q_table.values())}")
    
    # Train for 900 episodes
    print(f"\n✓ Training for 900 episodes (this may take 1-2 minutes)...")
    agent.train(episodes=900, max_hops=50)
    print(f"  ✓ Training complete!")
    
    # Compute statistics
    avg_reward = sum(agent.episode_rewards) / len(agent.episode_rewards)
    print(f"  Average episode reward: {avg_reward:.4f}")
    print(f"  Max episode reward: {max(agent.episode_rewards):.4f}")
    print(f"  Min episode reward: {min(agent.episode_rewards):.4f}")
    
    # ========================================================================
    # PART 3: Export Results
    # ========================================================================
    print("\n[STEP 3] Exporting Results")
    print("-" * 70)
    
    agent.export_q_table("outputs/q_table.csv")
    agent.export_training_log("outputs/training_log.csv")
    print(f"✓ Exported Q-table: outputs/q_table.csv")
    print(f"✓ Exported training log: outputs/training_log.csv")
    
    # ========================================================================
    # PART 4: Demonstrate Routing
    # ========================================================================
    print("\n[STEP 4] Demonstrating Routing")
    print("-" * 70)
    
    # Route from multiple sources
    for source_id in [0, 5, 10]:
        path, delay = agent.route_packet(source_id, destination_id=-1, max_hops=50)
        
        route_str = " → ".join([str(n) if n != -1 else "SINK" for n in path])
        print(f"\nRoute from node {source_id}:")
        print(f"  Path: {route_str}")
        print(f"  Hops: {len(path) - 1}")
        print(f"  Delay: {delay:.4f} sec")
        
        # Visualize this route
        if source_id == 0:
            visualizer = NetworkVisualizer(network)
            visualizer.plot_routing_path(path, f"outputs/route_from_{source_id}.png")
    
    # ========================================================================
    # PART 5: Analyze Q-Learning Convergence
    # ========================================================================
    print("\n[STEP 5] Analyzing Q-Learning Convergence")
    print("-" * 70)
    
    # Compute moving average
    window = 50
    moving_avg = []
    for i in range(window, len(agent.episode_rewards)):
        avg = sum(agent.episode_rewards[i-window:i]) / window
        moving_avg.append(avg)
    
    if moving_avg:
        print(f"✓ Moving average (50-episode window):")
        print(f"  Initial (ep 50-100): {moving_avg[0]:.4f}")
        print(f"  Mid (ep 450-500): {moving_avg[400]:.4f}")
        print(f"  Final (ep 850-900): {moving_avg[-1]:.4f}")
        
        convergence = abs(moving_avg[-1] - moving_avg[0])
        print(f"  Convergence change: {convergence:.4f}")
    
    # ========================================================================
    # PART 6: Network Statistics
    # ========================================================================
    print("\n[STEP 6] Final Network Statistics")
    print("-" * 70)
    
    stats = network.get_statistics()
    total_energy = stats['total_energy']
    print(f"✓ Network Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total energy: {total_energy:.2f} J")
    print(f"  Avg energy/node: {stats['avg_energy_per_node']:.2f} J")
    print(f"  Total packets sent: {stats['total_packets_sent']}")
    print(f"  Total packets received: {stats['total_packets_received']}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nGenerated Files:")
    print("  ✓ outputs/random_topology.png")
    print("  ✓ outputs/mesh_topology.png")
    print("  ✓ outputs/tree_topology.png")
    print("  ✓ outputs/q_table.csv")
    print("  ✓ outputs/training_log.csv")
    print("  ✓ outputs/route_from_0.png")
    print("\nNext Steps:")
    print("  1. Review outputs/ directory for visualizations")
    print("  2. Analyze Q-table.csv for learned policies")
    print("  3. Examine training_log.csv for convergence")
    print("  4. Modify epsilon, omega, or other parameters and retrain")
    print("  5. Run tests/smoke_test.py to verify implementation")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] User terminated execution")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
