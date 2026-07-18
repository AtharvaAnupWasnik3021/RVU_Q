"""
IEEE-conference-quality network topology visualization component.

Renders IoT mesh network topology with simulation state visualization:
- Node states: healthy, infected, exposed, sink, source, active routing, critical
- Edge states: healthy, active route, Q-learned route, failed, disconnected
- Smooth animations and visual effects (pulsing, glow, transitions)
- Optimized for 100-500 node networks
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon, FancyBboxPatch
from matplotlib.collections import LineCollection
from abc import ABC
from typing import Dict, List, Tuple, Optional, Any, Set
import warnings

from publication_viz_core import (
    VisualizationComponent, ColorPalette, Typography, smooth_step
)

warnings.filterwarnings('ignore')


# ============================================================================
# CONSTANTS
# ============================================================================

NODE_SIZE_MIN = 100
NODE_SIZE_MAX = 400
NODE_ALPHA = 0.85
EDGE_ALPHA_HEALTHY = 0.4
EDGE_ALPHA_DISCONNECTED = 0.15
EDGE_WIDTH_NORMAL = 1.0
EDGE_WIDTH_ACTIVE = 2.5
BORDER_WIDTH_HIGHLIGHT = 2.5
PULSING_DURATION = 20  # frames
GLOW_ITERATIONS = 3
ANIMATION_FRAMES = 10


# ============================================================================
# NETWORK RENDERER
# ============================================================================

class NetworkRenderer(VisualizationComponent):
    """
    Production-quality network topology visualization component.
    
    Renders IoT mesh network with node and edge state visualization,
    including smooth animations and visual effects. Optimized for
    100-500 node networks with proper scaling and alpha transparency.
    """
    
    def __init__(self, ax: plt.Axes, title: str = "Network Topology"):
        """
        Initialize network renderer.
        
        Args:
            ax: Matplotlib axes object
            title: Component title
        """
        super().__init__(ax, title)
        
        # Network structure
        self.node_positions: Dict[int, Tuple[float, float]] = {}
        self.node_patches: Dict[int, mpatches.Patch] = {}
        self.edge_lines: Dict[Tuple[int, int], Any] = {}
        
        # State tracking
        self.node_states: Dict[int, str] = {}  # node_id -> state
        self.edge_states: Dict[Tuple[int, int], str] = {}  # (src, dst) -> state
        self.animation_counters: Dict[int, int] = {}  # node_id -> frame count
        self.glow_patches: Dict[int, List[mpatches.Patch]] = {}  # node_id -> glow layers
        
        # Performance
        self.last_frame = -1
        self.frame_cache: Dict[int, Any] = {}
    
    def initialize(self):
        """
        Initialize component - set up static background network.
        Called once before animation starts.
        """
        self.set_title(self.title)
        
        # Configure axes for network visualization
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.08, color=ColorPalette.GRID_COLOR, linestyle=':')
        self.ax.set_facecolor(ColorPalette.BACKGROUND)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Set reasonable axis limits (will be adjusted based on network)
        self.ax.set_xlim(-1, 21)
        self.ax.set_ylim(-1, 21)
    
    def update(self, frame: int, state: Dict[str, Any]) -> List[Any]:
        """
        Update visualization for given frame.
        
        Args:
            frame: Frame number
            state: Simulation state dictionary containing:
                - 'nodes': Dict[int, Node]
                - 'edges': Dict[(int, int), str]
                - 'infected_nodes': Set[int]
                - 'exposed_nodes': Set[int]
                - 'active_route': List[int]
                - 'selected_qlearn_routes': Dict[int, List[int]]
                - 'failed_routes': Set[(int, int)]
        
        Returns:
            List of matplotlib artists for animation
        """
        artists = []
        
        # Extract state components
        nodes = state.get('nodes', {})
        edges = state.get('edges', {})
        infected = state.get('infected_nodes', set())
        exposed = state.get('exposed_nodes', set())
        active_route = state.get('active_route', [])
        qlearn_routes = state.get('selected_qlearn_routes', {})
        failed_routes = state.get('failed_routes', set())
        
        # Initialize network on first frame
        if frame == 0:
            self._draw_background_network(nodes, edges)
        
        # Update frame cache
        if frame != self.last_frame:
            self.last_frame = frame
            self._update_animation_counters()
        
        # Update states
        self._update_node_states(nodes, infected, exposed, active_route)
        self._update_edge_states(edges, active_route, qlearn_routes, failed_routes)
        
        # Apply visual effects
        self._add_node_effects(infected, exposed, frame)
        
        # Collect all artists
        artists.extend(self.node_patches.values())
        artists.extend(self.edge_lines.values())
        for glow_list in self.glow_patches.values():
            artists.extend(glow_list)
        
        return artists
    
    def _draw_background_network(self, nodes: Dict[int, Any],
                                 edges: Dict[Tuple[int, int], str]):
        """
        Draw all nodes and edges - called once during initialization.
        
        Args:
            nodes: Dictionary of node objects with x, y coordinates
            edges: Dictionary of edge types
        """
        if not nodes:
            return
        
        # Store node positions
        for node_id, node in nodes.items():
            self.node_positions[node_id] = (node.x, node.y)
            self.node_states[node_id] = 'healthy'
        
        # Update axis limits based on network bounds
        if self.node_positions:
            xs = [pos[0] for pos in self.node_positions.values()]
            ys = [pos[1] for pos in self.node_positions.values()]
            margin = 1.0
            self.ax.set_xlim(min(xs) - margin, max(xs) + margin)
            self.ax.set_ylim(min(ys) - margin, max(ys) + margin)
        
        # Draw edges first (so they appear behind nodes)
        for (src, dst), edge_type in edges.items():
            if src in self.node_positions and dst in self.node_positions:
                x1, y1 = self.node_positions[src]
                x2, y2 = self.node_positions[dst]
                
                edge_key = tuple(sorted([src, dst]))
                self.edge_states[edge_key] = 'healthy'
                
                # Draw edge
                line = self.ax.plot([x1, x2], [y1, y2],
                                   color=ColorPalette.HEALTHY_EDGE,
                                   alpha=EDGE_ALPHA_HEALTHY,
                                   linewidth=EDGE_WIDTH_NORMAL,
                                   zorder=1)[0]
                self.edge_lines[edge_key] = line
        
        # Draw nodes on top
        for node_id, (x, y) in self.node_positions.items():
            patch = self._create_node_patch(node_id, x, y, 'healthy')
            self.node_patches[node_id] = patch
            self.ax.add_patch(patch)
            
            # Initialize glow layers
            self.glow_patches[node_id] = []
    
    def _create_node_patch(self, node_id: int, x: float, y: float,
                          state: str) -> mpatches.Patch:
        """
        Create a node patch with appropriate styling.
        
        Args:
            node_id: Node identifier
            x, y: Node coordinates
            state: Node state ('healthy', 'infected', 'exposed', 'sink', 'source', etc.)
        
        Returns:
            Matplotlib patch object
        """
        # Determine size based on degree (node connectivity)
        degree = len([1 for (s, d) in self.edge_states.keys()
                     if s == node_id or d == node_id])
        size = NODE_SIZE_MIN + (degree / 10.0) * (NODE_SIZE_MAX - NODE_SIZE_MIN)
        radius = np.sqrt(size) / 100.0
        
        # Get color based on state
        color_map = {
            'healthy': ColorPalette.HEALTHY_NODE,
            'infected': ColorPalette.INFECTED_NODE,
            'exposed': ColorPalette.EXPOSED_NODE,
            'sink': ColorPalette.SINK_NODE,
            'source': ColorPalette.SOURCE_NODE,
        }
        color = color_map.get(state, ColorPalette.HEALTHY_NODE)
        
        # Create patch based on state
        if state == 'sink':
            # Draw as 5-pointed star
            patch = self._create_star_patch(x, y, radius, color)
        elif state == 'source':
            # Draw as triangle
            patch = self._create_triangle_patch(x, y, radius, color)
        else:
            # Draw as circle
            patch = Circle((x, y), radius, color=color, alpha=NODE_ALPHA,
                          edgecolor='none', zorder=10)
        
        return patch
    
    def _create_star_patch(self, x: float, y: float, radius: float,
                          color: str) -> Polygon:
        """
        Create a 5-pointed star polygon.
        
        Args:
            x, y: Center coordinates
            radius: Star radius
            color: Fill color
        
        Returns:
            Polygon patch
        """
        angles = np.linspace(0, 2*np.pi, 6)
        outer_r = radius
        inner_r = radius * 0.4
        vertices = []
        for i, angle in enumerate(angles[:-1]):
            r = outer_r if i % 2 == 0 else inner_r
            vertices.append([x + r * np.cos(angle), y + r * np.sin(angle)])
        return Polygon(vertices, color=color, alpha=NODE_ALPHA,
                      edgecolor=ColorPalette.TEXT_PRIMARY, linewidth=1.0, zorder=10)
    
    def _create_triangle_patch(self, x: float, y: float, radius: float,
                              color: str) -> Polygon:
        """
        Create a triangle polygon.
        
        Args:
            x, y: Center coordinates
            radius: Triangle radius
            color: Fill color
        
        Returns:
            Polygon patch
        """
        vertices = [
            [x, y + radius],
            [x - radius, y - radius/2],
            [x + radius, y - radius/2]
        ]
        return Polygon(vertices, color=color, alpha=NODE_ALPHA,
                      edgecolor=ColorPalette.TEXT_PRIMARY, linewidth=1.0, zorder=10)
    
    def _update_node_states(self, nodes: Dict[int, Any],
                           infected: Set[int], exposed: Set[int],
                           active_route: List[int]):
        """
        Update node colors based on infection status and routing state.
        
        Args:
            nodes: Dictionary of node objects
            infected: Set of infected node IDs
            exposed: Set of exposed node IDs
            active_route: List of nodes in current active route
        """
        for node_id, node in nodes.items():
            # Determine primary state
            if node_id in infected:
                new_state = 'infected'
            elif node_id in exposed:
                new_state = 'exposed'
            elif node.is_sink:
                new_state = 'sink'
            else:
                new_state = 'healthy'
            
            # Update state if changed
            if node_id in self.node_states and self.node_states[node_id] != new_state:
                self.node_states[node_id] = new_state
                
                # Replace node patch
                if node_id in self.node_patches:
                    old_patch = self.node_patches[node_id]
                    old_patch.remove()
                
                x, y = self.node_positions[node_id]
                new_patch = self._create_node_patch(node_id, x, y, new_state)
                self.ax.add_patch(new_patch)
                self.node_patches[node_id] = new_patch
            
            # Add routing highlights
            self._update_node_borders(node_id, active_route)
    
    def _update_node_borders(self, node_id: int, active_route: List[int]):
        """
        Add border highlights for special node states.
        
        Args:
            node_id: Node identifier
            active_route: List of nodes in active routing path
        """
        if node_id not in self.node_patches:
            return
        
        patch = self.node_patches[node_id]
        
        # Active routing nodes get cyan border
        if node_id in active_route and node_id not in [0]:  # Skip sink
            patch.set_edgecolor(ColorPalette.ACTIVE_ROUTING)
            patch.set_linewidth(BORDER_WIDTH_HIGHLIGHT)
        # Critical nodes get orange border (high infection risk)
        elif self.node_states.get(node_id) in ['infected', 'exposed']:
            patch.set_edgecolor(ColorPalette.CRITICAL_NODE)
            patch.set_linewidth(BORDER_WIDTH_HIGHLIGHT * 0.8)
        else:
            patch.set_edgecolor('none')
    
    def _update_edge_states(self, edges: Dict[Tuple[int, int], str],
                           active_route: List[int],
                           qlearn_routes: Dict[int, List[int]],
                           failed_routes: Set[Tuple[int, int]]):
        """
        Update edge colors and styles based on routing status.
        
        Args:
            edges: Dictionary of all edges and their types
            active_route: Current active route (path)
            qlearn_routes: Q-learned routes by destination
            failed_routes: Set of failed edge tuples
        """
        # Get active route edges
        active_route_edges = set()
        if len(active_route) > 1:
            for i in range(len(active_route) - 1):
                edge = tuple(sorted([active_route[i], active_route[i+1]]))
                active_route_edges.add(edge)
        
        # Get Q-learned route edges
        qlearn_edges = set()
        for dest, route in qlearn_routes.items():
            if len(route) > 1:
                for i in range(len(route) - 1):
                    edge = tuple(sorted([route[i], route[i+1]]))
                    qlearn_edges.add(edge)
        
        # Update all edges
        for edge_key in self.edge_lines.keys():
            # Determine edge state
            if edge_key in failed_routes or tuple(reversed(edge_key)) in failed_routes:
                new_state = 'failed'
                color = ColorPalette.FAILED_EDGE
                width = EDGE_WIDTH_NORMAL
                linestyle = '--'
                alpha = 0.7
            elif edge_key in active_route_edges:
                new_state = 'active'
                color = ColorPalette.ACTIVE_EDGE
                width = EDGE_WIDTH_ACTIVE
                linestyle = '-'
                alpha = 0.9
            elif edge_key in qlearn_edges:
                new_state = 'qlearn'
                color = ColorPalette.QLEARN_EDGE
                width = EDGE_WIDTH_NORMAL * 1.3
                linestyle = '-'
                alpha = 0.75
            else:
                new_state = 'healthy'
                color = ColorPalette.HEALTHY_EDGE
                width = EDGE_WIDTH_NORMAL
                linestyle = '-'
                alpha = EDGE_ALPHA_HEALTHY
            
            # Apply updates
            line = self.edge_lines[edge_key]
            line.set_color(color)
            line.set_linewidth(width)
            line.set_linestyle(linestyle)
            line.set_alpha(alpha)
            line.set_zorder(2 if new_state == 'active' else 1)
            
            self.edge_states[edge_key] = new_state
    
    def _add_node_effects(self, infected: Set[int], exposed: Set[int],
                         frame: int):
        """
        Add pulsing and glow effects to nodes.
        
        Args:
            infected: Set of infected node IDs
            exposed: Set of exposed node IDs
            frame: Current frame number
        """
        special_nodes = infected | exposed
        
        for node_id in special_nodes:
            if node_id not in self.node_patches:
                continue
            
            patch = self.node_patches[node_id]
            
            # Pulsing effect for infected nodes
            if node_id in infected:
                self._add_node_pulsing_effect(patch, frame)
            
            # Glow effect for all special nodes
            if node_id in special_nodes:
                self._add_node_glow(node_id, frame)
    
    def _add_node_pulsing_effect(self, patch: mpatches.Patch, frame: int):
        """
        Add pulsating effect to a node.
        
        Args:
            patch: Node patch object
            frame: Current frame number
        """
        # Pulse using sine wave (oscillate between 0.6 and 1.0 of base alpha)
        pulse_phase = (frame % PULSING_DURATION) / PULSING_DURATION
        pulse_factor = 0.8 + 0.2 * np.sin(pulse_phase * 2 * np.pi)
        
        # Modulate alpha transparency (clamped to valid range)
        if isinstance(patch, Circle):
            base_alpha = NODE_ALPHA
            new_alpha = np.clip(base_alpha * pulse_factor, 0.0, 1.0)
            patch.set_alpha(new_alpha)
    
    def _add_node_glow(self, node_id: int, frame: int):
        """
        Add glow effect to infected/critical nodes.
        
        Args:
            node_id: Node identifier
            frame: Current frame number
        """
        if node_id not in self.node_positions:
            return
        
        x, y = self.node_positions[node_id]
        
        # Remove old glow layers
        for old_patch in self.glow_patches[node_id]:
            old_patch.remove()
        self.glow_patches[node_id] = []
        
        # Create new glow layers
        base_radius = 0.15
        glow_color = ColorPalette.INFECTED_NODE
        
        for i in range(GLOW_ITERATIONS):
            alpha_glow = 0.15 * (1 - i / GLOW_ITERATIONS)
            radius_glow = base_radius * (1 + i * 0.4)
            
            glow_circle = Circle((x, y), radius_glow, color=glow_color,
                                alpha=alpha_glow, edgecolor='none', zorder=5)
            self.ax.add_patch(glow_circle)
            self.glow_patches[node_id].append(glow_circle)
    
    def _update_animation_counters(self):
        """Update frame counters for animation effects."""
        for node_id in self.animation_counters:
            self.animation_counters[node_id] = (
                self.animation_counters[node_id] + 1) % PULSING_DURATION


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_network_visualization(fig_width: float = 10,
                                fig_height: float = 10) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure and axes for network visualization.
    
    Args:
        fig_width: Figure width in inches
        fig_height: Figure height in inches
    
    Returns:
        Tuple of (figure, axes)
    """
    Typography.configure_mpl()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.set_facecolor(ColorPalette.BACKGROUND)
    return fig, ax


def get_node_color(state: str) -> str:
    """
    Get color for a node state.
    
    Args:
        state: Node state string
    
    Returns:
        Color hex code
    """
    color_map = {
        'healthy': ColorPalette.HEALTHY_NODE,
        'infected': ColorPalette.INFECTED_NODE,
        'exposed': ColorPalette.EXPOSED_NODE,
        'sink': ColorPalette.SINK_NODE,
        'source': ColorPalette.SOURCE_NODE,
        'active': ColorPalette.ACTIVE_ROUTING,
        'critical': ColorPalette.CRITICAL_NODE,
    }
    return color_map.get(state, ColorPalette.HEALTHY_NODE)


def get_edge_color(state: str) -> str:
    """
    Get color for an edge state.
    
    Args:
        state: Edge state string
    
    Returns:
        Color hex code
    """
    color_map = {
        'healthy': ColorPalette.HEALTHY_EDGE,
        'active': ColorPalette.ACTIVE_EDGE,
        'qlearn': ColorPalette.QLEARN_EDGE,
        'failed': ColorPalette.FAILED_EDGE,
        'disconnected': ColorPalette.DISCONNECTED_EDGE,
    }
    return color_map.get(state, ColorPalette.HEALTHY_EDGE)
