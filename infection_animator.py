"""
IEEE-conference-quality infection spread animation component for IoT visualization.

Animates realistic infection propagation through network with:
- Multi-state node transitions: Healthy (green) → Exposed (orange) → Infected (red)
- Smooth color transitions over 5-10 frames per state change
- Pulsating red glow effect for fully infected nodes
- Visual "wave" of infection spreading through neighboring nodes
- Infection state tracking to show progression

Production-ready with proper docstrings, error handling, and smooth visual
effects at 30+ FPS.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings

from publication_viz_core import (
    VisualizationComponent, ColorPalette, Typography, smooth_step, lerp
)

warnings.filterwarnings('ignore')


# ============================================================================
# CONSTANTS
# ============================================================================

# State machine constants
class InfectionState(Enum):
    """Node infection states."""
    HEALTHY = "healthy"
    EXPOSED = "exposed"
    INFECTED = "infected"


# Animation parameters
FRAMES_PER_TRANSITION = 7  # Frames for smooth transition between states
PULSE_FREQUENCY = 1.0  # Hz (cycles per second)
PULSE_AMPLITUDE = 0.2  # 20% brightness change
GLOW_OUTER_ALPHA = 0.15  # Glow outer ring transparency
GLOW_INNER_ALPHA = 0.3  # Glow inner ring transparency
GLOW_SCALE = 1.5  # Glow size multiplier
WAVE_DURATION = 30  # Frames for infection wave animation
WAVE_ALPHA_MAX = 0.4  # Maximum alpha for wave circle
MIN_NODE_SIZE = 30  # Minimum node size
MAX_NODE_SIZE = 120  # Maximum node size (for pulsing)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class NodeInfectionState:
    """Tracks infection state for a single node."""
    node_id: int
    current_state: InfectionState = InfectionState.HEALTHY
    transition_progress: float = 0.0  # 0-1 during state transitions
    frames_in_state: int = 0  # Frames spent in current state
    is_pulsing: bool = False  # Whether node should pulse
    glow_intensity: float = 0.0  # Current glow intensity (0-1)
    last_infection_frame: int = -100  # Frame when last infected
    
    def get_current_color(self, palette: Dict[str, str]) -> str:
        """
        Get current node color based on state and transition progress.
        
        Args:
            palette: Color palette dict with states as keys
        
        Returns:
            Hex color string
        """
        if self.current_state == InfectionState.HEALTHY:
            return palette.get('healthy', ColorPalette.HEALTHY_NODE)
        elif self.current_state == InfectionState.EXPOSED:
            return palette.get('exposed', ColorPalette.EXPOSED_NODE)
        elif self.current_state == InfectionState.INFECTED:
            return palette.get('infected', ColorPalette.INFECTED_NODE)
        return palette.get('healthy', ColorPalette.HEALTHY_NODE)


@dataclass
class InfectionWave:
    """Represents an expanding infection wave."""
    source_node_id: int
    center_x: float
    center_y: float
    creation_frame: int
    age_frames: int = 0  # How many frames old
    
    def is_active(self, current_frame: int) -> bool:
        """Check if wave should still be rendered."""
        self.age_frames = current_frame - self.creation_frame
        return self.age_frames < WAVE_DURATION


# ============================================================================
# INFECTION ANIMATOR COMPONENT
# ============================================================================

class InfectionAnimator(VisualizationComponent):
    """
    Production-quality infection animation component for IoT network visualization.
    
    Visualizes realistic infection propagation through network with multi-state
    transitions, smooth color animations, pulsating effects, and wave visualization.
    Efficiently handles network-scale simulations at 30+ FPS.
    
    Integration with simulation:
    - Expects state dict with 'nodes', 'infection_levels', 'exposed_nodes', 'newly_infected'
    - Automatically synchronizes with simulation infection state
    - Tracks per-node infection state and transition progress
    """
    
    def __init__(self, ax: plt.Axes, title: str = "Infection Spread Animation"):
        """
        Initialize infection animator component.
        
        Args:
            ax: Matplotlib axes object
            title: Component title
        """
        super().__init__(ax, title)
        
        # Infection state tracking
        self.node_states: Dict[int, NodeInfectionState] = {}
        self.infection_waves: List[InfectionWave] = []
        self.node_positions: Dict[int, Tuple[float, float]] = {}
        self.node_artists: Dict[int, Circle] = {}
        self.glow_artists: Dict[int, List[Any]] = {}
        
        # Animation state
        self.current_frame = 0
        self.time_accumulator = 0.0
        self.last_infection_count = 0
        
        # Color palette
        self.color_palette = {
            'healthy': ColorPalette.HEALTHY_NODE,
            'exposed': ColorPalette.EXPOSED_NODE,
            'infected': ColorPalette.INFECTED_NODE,
        }
    
    def initialize(self):
        """
        Initialize component - set up infection animation system.
        Called once before animation starts.
        """
        self.set_title(self.title)
        
        # Configure axes
        self.ax.set_aspect('equal')
        self.ax.set_facecolor(ColorPalette.BACKGROUND)
    
    def update(self, frame: int, state: Dict[str, Any]) -> List[Any]:
        """
        Update infection animation for given frame.
        
        Args:
            frame: Frame number
            state: Simulation state dictionary containing:
                - 'nodes': Dict[int, Node] (nodes with 'infected' property)
                - 'infection_levels': List[int] (infection progression levels)
                - 'exposed_nodes': Set[int] (about to be infected)
                - 'newly_infected': Set[int] (just became infected)
                - 'node_positions': Dict[int, Tuple[float, float]]
                - 'edges': Optional list of edges for topology
        
        Returns:
            List of matplotlib artists for animation
        """
        artists = []
        
        # Extract state components
        nodes = state.get('nodes', {})
        exposed_nodes = state.get('exposed_nodes', set())
        newly_infected = state.get('newly_infected', set())
        self.node_positions = state.get('node_positions', {})
        
        self.current_frame = frame
        self.time_accumulator = frame / 30.0  # Assume 30 FPS
        
        # Update node infection states based on simulation
        self._update_node_infection_states(
            nodes, exposed_nodes, newly_infected
        )
        
        # Update existing nodes and create new ones
        self._update_node_visuals()
        
        # Create waves for newly infected nodes
        self._create_infection_waves(newly_infected)
        
        # Update and draw infection waves
        self._render_infection_waves()
        
        # Apply smooth color transitions
        self._apply_infection_transitions()
        
        # Add pulsating effects to infected nodes
        self._add_infection_pulsing()
        
        # Add glow effects around infected nodes
        self._add_infection_glow()
        
        # Collect all artists
        artists.extend(self.node_artists.values())
        for glow_list in self.glow_artists.values():
            artists.extend(glow_list)
        
        return artists
    
    def _update_node_infection_states(self, nodes: Dict[int, Any],
                                     exposed_nodes: Set[int],
                                     newly_infected: Set[int]):
        """
        Update per-node infection state tracking.
        
        Manages state machine transitions: HEALTHY → EXPOSED → INFECTED.
        Tracks transition progress for smooth animations.
        
        Args:
            nodes: Dict of node objects from simulation
            exposed_nodes: Set of node IDs in exposed state
            newly_infected: Set of node IDs just infected this frame
        """
        # Initialize states for any new nodes
        for node_id in nodes.keys():
            if node_id not in self.node_states:
                self.node_states[node_id] = NodeInfectionState(node_id=node_id)
        
        # Update states based on simulation
        for node_id, node_state in self.node_states.items():
            if node_id not in nodes:
                continue
            
            node = nodes[node_id]
            is_infected = getattr(node, 'infected', False)
            
            # State transitions
            if is_infected:
                if node_state.current_state != InfectionState.INFECTED:
                    # Transition to infected
                    node_state.current_state = InfectionState.INFECTED
                    node_state.transition_progress = 0.0
                    node_state.frames_in_state = 0
                    node_state.is_pulsing = True
                    node_state.last_infection_frame = self.current_frame
                else:
                    # Already infected
                    node_state.frames_in_state += 1
                    node_state.is_pulsing = True
                    # Progress toward full infection
                    node_state.transition_progress = min(
                        1.0,
                        node_state.transition_progress + 1.0 / FRAMES_PER_TRANSITION
                    )
            
            elif node_id in exposed_nodes:
                if node_state.current_state != InfectionState.EXPOSED:
                    # Transition to exposed
                    node_state.current_state = InfectionState.EXPOSED
                    node_state.transition_progress = 0.0
                    node_state.frames_in_state = 0
                    node_state.is_pulsing = False
                else:
                    # Already exposed
                    node_state.frames_in_state += 1
                    # Progress through exposed state
                    node_state.transition_progress = min(
                        1.0,
                        node_state.transition_progress + 1.0 / FRAMES_PER_TRANSITION
                    )
            
            else:
                # Healthy state (or transitioning back)
                if node_state.current_state != InfectionState.HEALTHY:
                    # Transition to healthy
                    node_state.current_state = InfectionState.HEALTHY
                    node_state.transition_progress = 0.0
                    node_state.frames_in_state = 0
                    node_state.is_pulsing = False
    
    def _update_node_visuals(self):
        """
        Create or update visual representations of nodes.
        
        Creates CirclePatches for each node with appropriate colors
        and positions from simulation state.
        """
        for node_id, position in self.node_positions.items():
            if node_id not in self.node_states:
                continue
            
            state = self.node_states[node_id]
            x, y = position
            
            # Get base node size
            size_radius = np.sqrt(MIN_NODE_SIZE) / 100.0
            
            # Get current color
            color = state.get_current_color(self.color_palette)
            
            # Remove old artist if exists
            if node_id in self.node_artists:
                self.node_artists[node_id].remove()
            
            # Create new node circle
            circle = Circle(
                (x, y),
                size_radius,
                color=color,
                alpha=0.9,
                zorder=10
            )
            self.ax.add_patch(circle)
            self.node_artists[node_id] = circle
    
    def _create_infection_waves(self, newly_infected: Set[int]):
        """
        Create new infection waves from recently infected nodes.
        
        Args:
            newly_infected: Set of node IDs just infected
        """
        for node_id in newly_infected:
            if node_id in self.node_positions:
                x, y = self.node_positions[node_id]
                wave = InfectionWave(
                    source_node_id=node_id,
                    center_x=x,
                    center_y=y,
                    creation_frame=self.current_frame
                )
                self.infection_waves.append(wave)
    
    def _render_infection_waves(self):
        """
        Draw expanding infection waves for newly infected nodes.
        
        Renders expanding circles that fade out over time to show
        infection spreading pattern.
        """
        # Remove old wave artists
        expired_waves = []
        
        for i, wave in enumerate(self.infection_waves):
            if not wave.is_active(self.current_frame):
                expired_waves.append(i)
                continue
            
            # Calculate wave properties
            progress = wave.age_frames / WAVE_DURATION
            alpha = WAVE_ALPHA_MAX * (1.0 - progress)  # Fade out
            
            # Wave expands outward
            max_radius = 0.4  # Expands to this radius
            radius = max_radius * smooth_step(progress)
            
            # Draw wave
            if radius > 0.01:
                wave_circle = Circle(
                    (wave.center_x, wave.center_y),
                    radius,
                    fill=False,
                    edgecolor=ColorPalette.INFECTED_NODE,
                    linewidth=2,
                    alpha=alpha,
                    zorder=5
                )
                self.ax.add_patch(wave_circle)
                self.artists.append(wave_circle)
        
        # Remove expired waves
        for i in reversed(expired_waves):
            self.infection_waves.pop(i)
    
    def _apply_infection_transitions(self):
        """
        Apply smooth color transitions between infection states.
        
        Interpolates node colors based on transition progress using
        smooth_step for easing.
        """
        for node_id, node_state in self.node_states.items():
            if node_id not in self.node_artists:
                continue
            
            circle = self.node_artists[node_id]
            
            # Use smooth_step for easing
            eased_progress = smooth_step(node_state.transition_progress)
            
            # Get source and target colors based on state
            if node_state.current_state == InfectionState.EXPOSED:
                # Transitioning from healthy to exposed
                src_color = ColorPalette.HEALTHY_NODE
                dst_color = ColorPalette.EXPOSED_NODE
                color = self._interpolate_color(
                    src_color, dst_color, eased_progress
                )
            
            elif node_state.current_state == InfectionState.INFECTED:
                # Transitioning from exposed/healthy to infected
                if node_state.frames_in_state == 0:
                    src_color = ColorPalette.EXPOSED_NODE
                else:
                    src_color = ColorPalette.INFECTED_NODE
                dst_color = ColorPalette.INFECTED_NODE
                color = self._interpolate_color(
                    src_color, dst_color, eased_progress
                )
            
            else:
                # Healthy state
                color = node_state.get_current_color(self.color_palette)
            
            # Update circle color
            circle.set_color(color)
    
    def _add_infection_pulsing(self):
        """
        Add pulsating effect to infected nodes.
        
        Modulates node size with sinusoidal function at PULSE_FREQUENCY.
        Creates visual "heartbeat" effect for infected nodes.
        """
        for node_id, node_state in self.node_states.items():
            if not node_state.is_pulsing or node_id not in self.node_artists:
                continue
            
            circle = self.node_artists[node_id]
            
            # Calculate pulsation
            phase = 2 * np.pi * PULSE_FREQUENCY * self.time_accumulator
            pulse = 1.0 + PULSE_AMPLITUDE * np.sin(phase)
            
            # Modulate size
            base_radius = np.sqrt(MIN_NODE_SIZE) / 100.0
            new_radius = base_radius * pulse
            circle.set_radius(new_radius)
            
            # Modulate alpha for breathing effect
            base_alpha = 0.9
            alpha = base_alpha * (1.0 + 0.3 * np.sin(phase + np.pi/2))
            circle.set_alpha(np.clip(alpha, 0.6, 1.0))
    
    def _add_infection_glow(self):
        """
        Add glowing aura effect around infected nodes.
        
        Renders concentric semi-transparent circles around infected nodes
        to create a glow effect that intensifies with infection progress.
        """
        # Remove old glow artists
        for old_glows in self.glow_artists.values():
            for glow in old_glows:
                if glow in self.ax.patches:
                    glow.remove()
        self.glow_artists.clear()
        
        # Add new glows for infected nodes
        for node_id, node_state in self.node_states.items():
            if node_state.current_state != InfectionState.INFECTED:
                continue
            
            if node_id not in self.node_positions:
                continue
            
            x, y = self.node_positions[node_id]
            base_radius = np.sqrt(MIN_NODE_SIZE) / 100.0
            
            # Glow intensity increases with time infected
            intensity = min(1.0, node_state.frames_in_state / 20.0)
            
            glows = []
            
            # Outer glow (faint)
            outer_radius = base_radius * GLOW_SCALE * (1.2 + 0.3 * intensity)
            outer_glow = Circle(
                (x, y),
                outer_radius,
                color=ColorPalette.INFECTED_NODE,
                alpha=GLOW_OUTER_ALPHA * intensity,
                zorder=8
            )
            self.ax.add_patch(outer_glow)
            glows.append(outer_glow)
            
            # Inner glow (more intense)
            inner_radius = base_radius * GLOW_SCALE * (1.1 + 0.2 * intensity)
            inner_glow = Circle(
                (x, y),
                inner_radius,
                color=ColorPalette.INFECTED_NODE,
                alpha=GLOW_INNER_ALPHA * intensity,
                zorder=9
            )
            self.ax.add_patch(inner_glow)
            glows.append(inner_glow)
            
            self.glow_artists[node_id] = glows
    
    def _interpolate_color(self, color1: str, color2: str, 
                          t: float) -> str:
        """
        Interpolate between two hex colors.
        
        Args:
            color1: Starting color as hex string
            color2: Ending color as hex string
            t: Interpolation parameter (0-1)
        
        Returns:
            Interpolated color as hex string
        """
        # Parse hex colors
        try:
            c1 = self._hex_to_rgb(color1)
            c2 = self._hex_to_rgb(color2)
        except (ValueError, IndexError):
            return color1
        
        # Interpolate each channel
        r = int(lerp(c1[0], c2[0], t))
        g = int(lerp(c1[1], c2[1], t))
        b = int(lerp(c1[2], c2[2], t))
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """
        Convert hex color string to RGB tuple.
        
        Args:
            hex_color: Color as hex string (e.g., '#FF0000')
        
        Returns:
            Tuple of (R, G, B) values 0-255
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def clear(self):
        """
        Clear all infection animation artists from the axis.
        Extends base clear method to handle infection-specific artists.
        """
        # Clear wave artists
        for wave in self.infection_waves:
            # Waves are added to self.artists, so they'll be cleared
            pass
        
        # Clear node artists
        for node_circle in self.node_artists.values():
            if node_circle in self.ax.patches:
                node_circle.remove()
        
        # Clear glow artists
        for glow_list in self.glow_artists.values():
            for glow in glow_list:
                if glow in self.ax.patches:
                    glow.remove()
        
        self.node_artists.clear()
        self.glow_artists.clear()
        self.infection_waves.clear()
        
        # Call parent clear
        super().clear()
