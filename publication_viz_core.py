"""
Publication-quality visualization core utilities and base classes.

Provides shared infrastructure for all visualization components:
- Color schemes and styling (IEEE/ACM/Nature ML standards)
- Layout grid management
- Base animation classes
- Typography and formatting utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon, Wedge, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# COLOR SCHEMES - IEEE Conference Quality
# ============================================================================

class ColorPalette:
    """Professional color palette for research visualization."""
    
    # Dark theme background
    BACKGROUND = '#0a0e27'      # Very dark blue-gray
    GRID_COLOR = '#1a1f3a'      # Subtle grid
    TEXT_PRIMARY = '#e8e8e8'    # Off-white text
    TEXT_SECONDARY = '#b0b0b0'  # Gray text
    
    # Node states
    HEALTHY_NODE = '#4CAF50'       # Fresh green
    INFECTED_NODE = '#FF3D3D'      # Bright red
    SINK_NODE = '#2196F3'          # Vibrant blue
    SOURCE_NODE = '#FFD700'        # Gold yellow
    ACTIVE_ROUTING = '#00BCD4'     # Cyan (border)
    CRITICAL_NODE = '#FF9800'      # Orange (border)
    EXPOSED_NODE = '#FFA726'       # Light orange
    
    # Edge states
    HEALTHY_EDGE = '#424242'       # Dark gray
    ACTIVE_EDGE = '#00E5FF'        # Bright cyan (thicker)
    QLEARN_EDGE = '#00C853'        # Bright green glow
    FAILED_EDGE = '#FF5252'        # Bright red (dashed)
    DISCONNECTED_EDGE = '#9E9E9E'  # Light gray (faded)
    
    # Packet states
    PACKET_NORMAL = '#64B5F6'      # Light blue
    PACKET_DELIVERED = '#81C784'   # Light green
    PACKET_DROPPED = '#EF5350'     # Light red
    PACKET_REROUTED = '#FFD54F'    # Light yellow
    
    # Dashboard metrics
    REWARD_POSITIVE = '#4CAF50'
    REWARD_NEGATIVE = '#FF3D3D'
    REWARD_NEUTRAL = '#9E9E9E'
    
    # Protocol colors
    PROTOCOL_QLEARNING = '#2196F3'  # Blue
    PROTOCOL_DIJKSTRA = '#4CAF50'   # Green
    PROTOCOL_AODV = '#FF9800'       # Orange
    PROTOCOL_FLOODING = '#FF3D3D'   # Red
    
    @staticmethod
    def get_protocol_color(protocol: str) -> str:
        """Get color for a protocol name."""
        colors = {
            'q-learning': ColorPalette.PROTOCOL_QLEARNING,
            'dijkstra': ColorPalette.PROTOCOL_DIJKSTRA,
            'aodv': ColorPalette.PROTOCOL_AODV,
            'flooding': ColorPalette.PROTOCOL_FLOODING,
        }
        return colors.get(protocol.lower(), ColorPalette.TEXT_PRIMARY)
    
    @staticmethod
    def reward_color(reward: float, min_reward: float = -10, max_reward: float = 20) -> str:
        """Get color based on reward value (gradient from red to green)."""
        normalized = (reward - min_reward) / (max_reward - min_reward)
        normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
        
        if normalized < 0.5:
            # Red to yellow
            r_val = 1.0
            g_val = normalized * 2  # 0 to 1
            b_val = 0
        else:
            # Yellow to green
            r_val = 1 - (normalized - 0.5) * 2  # 1 to 0
            g_val = 1.0
            b_val = 0
        
        return f'#{int(r_val*255):02x}{int(g_val*255):02x}{int(b_val*255):02x}'


# ============================================================================
# TYPOGRAPHY AND STYLING
# ============================================================================

class Typography:
    """Publication-quality typography settings."""
    
    # Font families
    FONT_SANS = 'sans-serif'
    
    # Font sizes (points)
    SIZE_TITLE = 20
    SIZE_SECTION = 14
    SIZE_LABEL = 11
    SIZE_SMALL = 9
    
    # Font weights
    WEIGHT_BOLD = 'bold'
    WEIGHT_NORMAL = 'normal'
    
    @staticmethod
    def configure_mpl():
        """Configure matplotlib for publication quality."""
        plt.rcParams['figure.facecolor'] = ColorPalette.BACKGROUND
        plt.rcParams['axes.facecolor'] = ColorPalette.BACKGROUND
        plt.rcParams['axes.edgecolor'] = ColorPalette.GRID_COLOR
        plt.rcParams['axes.labelcolor'] = ColorPalette.TEXT_PRIMARY
        plt.rcParams['axes.titlecolor'] = ColorPalette.TEXT_PRIMARY
        plt.rcParams['xtick.color'] = ColorPalette.TEXT_SECONDARY
        plt.rcParams['ytick.color'] = ColorPalette.TEXT_SECONDARY
        plt.rcParams['text.color'] = ColorPalette.TEXT_PRIMARY
        plt.rcParams['font.size'] = Typography.SIZE_LABEL
        plt.rcParams['legend.facecolor'] = ColorPalette.GRID_COLOR
        plt.rcParams['legend.edgecolor'] = ColorPalette.GRID_COLOR
        plt.rcParams['legend.framealpha'] = 0.9
        plt.rcParams['lines.antialiased'] = True
        plt.rcParams['patch.antialiased'] = True
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150


# ============================================================================
# LAYOUT GRID MANAGEMENT
# ============================================================================

@dataclass
class GridPosition:
    """Position in a grid layout."""
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    
    def to_gridspec_coords(self) -> Tuple[int, int, int, int]:
        """Convert to (row_start, row_end, col_start, col_end)."""
        return (
            self.row,
            self.row + self.rowspan,
            self.col,
            self.col + self.colspan
        )


class LayoutManager:
    """Manages 2x2 grid layout for publication visualization."""
    
    def __init__(self, figsize: Tuple[int, int] = (1920//100, 1080//100),
                 dpi: int = 100):
        """
        Initialize layout manager.
        
        Args:
            figsize: Figure size in inches (default 19.2x10.8 for 1920x1080 @ 100 DPI)
            dpi: Dots per inch
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.axes = {}
        self.heights = [2, 1]  # Top takes 2x space, bottom takes 1x
        self.widths = [1.2, 0.8]  # Left takes 1.2x space, right takes 0.8x
    
    def create_layout(self) -> Dict[str, plt.Axes]:
        """
        Create figure with 2x2 grid layout.
        
        Returns:
            Dictionary mapping position names to axes:
            - 'network': Top-left (main network)
            - 'rl_dashboard': Top-right
            - 'metrics': Bottom-left
            - 'protocol_comparison': Bottom-right
        """
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.fig.patch.set_facecolor(ColorPalette.BACKGROUND)
        
        # Create grid
        gs = self.fig.add_gridspec(
            2, 2,
            height_ratios=self.heights,
            width_ratios=self.widths,
            hspace=0.25,
            wspace=0.25
        )
        
        # Create axes
        ax_network = self.fig.add_subplot(gs[0, 0])
        ax_rl = self.fig.add_subplot(gs[0, 1])
        ax_metrics = self.fig.add_subplot(gs[1, 0])
        ax_protocol = self.fig.add_subplot(gs[1, 1])
        
        self.axes = {
            'network': ax_network,
            'rl_dashboard': ax_rl,
            'metrics': ax_metrics,
            'protocol_comparison': ax_protocol,
        }
        
        # Configure all axes
        for name, ax in self.axes.items():
            self._configure_axis(ax, name)
        
        return self.axes
    
    def _configure_axis(self, ax: plt.Axes, name: str):
        """Configure axis styling."""
        ax.set_facecolor(ColorPalette.BACKGROUND)
        ax.spines['bottom'].set_color(ColorPalette.GRID_COLOR)
        ax.spines['top'].set_color(ColorPalette.GRID_COLOR)
        ax.spines['right'].set_color(ColorPalette.GRID_COLOR)
        ax.spines['left'].set_color(ColorPalette.GRID_COLOR)
        ax.tick_params(colors=ColorPalette.TEXT_SECONDARY)
        
        if name == 'network':
            # Network plot: remove ticks and labels, keep grid subtle
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.1, color=ColorPalette.GRID_COLOR)
        else:
            ax.grid(True, alpha=0.15, color=ColorPalette.GRID_COLOR)
    
    def get_figure(self) -> plt.Figure:
        """Get the figure object."""
        return self.fig


# ============================================================================
# BASE ANIMATION COMPONENT
# ============================================================================

class VisualizationComponent(ABC):
    """Base class for all visualization components."""
    
    def __init__(self, ax: plt.Axes, title: str = ""):
        """
        Initialize visualization component.
        
        Args:
            ax: Matplotlib axes object
            title: Component title
        """
        self.ax = ax
        self.title = title
        self.collections = []
        self.artists = []
    
    @abstractmethod
    def initialize(self):
        """Initialize component (called once before animation starts)."""
        pass
    
    @abstractmethod
    def update(self, frame: int, state: Dict[str, Any]) -> List[Any]:
        """
        Update component for given frame.
        
        Args:
            frame: Frame number
            state: Simulation state dictionary
        
        Returns:
            List of matplotlib artists to animate
        """
        pass
    
    def clear(self):
        """Clear all artists from the axis."""
        for artist in self.artists:
            artist.remove()
        for collection in self.collections:
            collection.remove()
        self.artists = []
        self.collections = []
    
    def set_title(self, title: str, **kwargs):
        """Set component title with styling."""
        self.ax.set_title(
            title,
            fontsize=Typography.SIZE_SECTION,
            fontweight=Typography.WEIGHT_BOLD,
            color=ColorPalette.TEXT_PRIMARY,
            pad=10,
            **kwargs
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def draw_node(ax: plt.Axes, x: float, y: float, node_type: str = 'healthy',
              size: float = 100, alpha: float = 1.0) -> mpatches.Patch:
    """
    Draw a node at (x, y) with styling based on type.
    
    Args:
        ax: Matplotlib axes
        x, y: Node coordinates
        node_type: One of 'healthy', 'infected', 'exposed', 'sink', 'source', 'active', 'critical'
        size: Node size in points
        alpha: Transparency
    
    Returns:
        Matplotlib patch object
    """
    colors = {
        'healthy': ColorPalette.HEALTHY_NODE,
        'infected': ColorPalette.INFECTED_NODE,
        'exposed': ColorPalette.EXPOSED_NODE,
        'sink': ColorPalette.SINK_NODE,
        'source': ColorPalette.SOURCE_NODE,
        'active': ColorPalette.ACTIVE_ROUTING,
        'critical': ColorPalette.CRITICAL_NODE,
    }
    
    color = colors.get(node_type, ColorPalette.HEALTHY_NODE)
    
    if node_type == 'sink':
        # Draw as star
        angles = np.linspace(0, 2*np.pi, 6)
        outer_r = np.sqrt(size) / 100
        inner_r = outer_r * 0.4
        vertices = []
        for i, angle in enumerate(angles):
            r = outer_r if i % 2 == 0 else inner_r
            vertices.append([x + r * np.cos(angle), y + r * np.sin(angle)])
        star = Polygon(vertices, color=color, alpha=alpha, ec='white', lw=1.5)
        ax.add_patch(star)
        return star
    
    elif node_type == 'source':
        # Draw as triangle
        triangle = Polygon(
            [[x, y + np.sqrt(size)/100], [x - np.sqrt(size)/100, y - np.sqrt(size)/200],
             [x + np.sqrt(size)/100, y - np.sqrt(size)/200]],
            color=color, alpha=alpha, ec='white', lw=1.5
        )
        ax.add_patch(triangle)
        return triangle
    
    else:
        # Draw as circle
        circle = Circle((x, y), np.sqrt(size)/100, color=color, alpha=alpha)
        if node_type in ['active', 'critical']:
            circle.set_ec('white')
            circle.set_linewidth(2)
        ax.add_patch(circle)
        return circle


def draw_edge(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float,
              edge_type: str = 'healthy', alpha: float = 0.6, linewidth: float = 1) -> Any:
    """
    Draw an edge between two points.
    
    Args:
        ax: Matplotlib axes
        x1, y1, x2, y2: Coordinates
        edge_type: One of 'healthy', 'active', 'qlearn', 'failed', 'disconnected'
        alpha: Transparency
        linewidth: Line width
    
    Returns:
        Matplotlib artist
    """
    colors = {
        'healthy': ColorPalette.HEALTHY_EDGE,
        'active': ColorPalette.ACTIVE_EDGE,
        'qlearn': ColorPalette.QLEARN_EDGE,
        'failed': ColorPalette.FAILED_EDGE,
        'disconnected': ColorPalette.DISCONNECTED_EDGE,
    }
    
    color = colors.get(edge_type, ColorPalette.HEALTHY_EDGE)
    
    if edge_type == 'active':
        linewidth = max(2, linewidth * 1.5)
    
    if edge_type == 'failed':
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth,
                linestyle='--', zorder=1)
    else:
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth,
                zorder=1)
    
    line = ax.lines[-1]
    return line


def format_metric(value: float, name: str, unit: str = "") -> str:
    """Format a metric value for display."""
    if name in ['pdr', 'connectivity', 'exploitation', 'security_score', 'node_survival']:
        return f"{value:.1f}%"
    elif name in ['reward', 'avg_reward', 'q_value_mean', 'q_value_variance']:
        return f"{value:.3f}"
    elif name in ['delay', 'energy', 'residual_energy']:
        return f"{value:.2f}{unit}"
    elif name in ['epsilon']:
        return f"{value:.4f}"
    else:
        return f"{value:.1f}" if isinstance(value, float) else str(value)


def smooth_step(t: float) -> float:
    """Smooth step function for animations (Hermite interpolation)."""
    t = max(0, min(1, t))
    return t * t * (3 - 2 * t)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


# ============================================================================
# INITIALIZATION
# ============================================================================

def setup_publication_style():
    """Call this once at script startup to configure matplotlib for publication."""
    Typography.configure_mpl()
