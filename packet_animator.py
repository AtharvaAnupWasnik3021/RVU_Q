"""
IEEE-conference-quality packet animation component for IoT network visualization.

Animates individual data packets moving along network routes with:
- Colored dots representing packet states (blue/green/red/yellow)
- Smooth frame-by-frame movement along paths
- Multiple simultaneous packets (5-50 on screen)
- Speed proportional to route length
- Motion trails for visual appeal
- Production-ready with proper docstrings, error handling, and optimization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection
from matplotlib.patches import Circle
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import warnings

from publication_viz_core import (
    VisualizationComponent, ColorPalette, Typography, smooth_step, lerp
)

warnings.filterwarnings('ignore')


# ============================================================================
# CONSTANTS
# ============================================================================

PACKET_SIZE_MIN = 8  # Minimum packet diameter in points
PACKET_SIZE_MAX = 20  # Maximum packet diameter in points
PACKET_TRAIL_LENGTH = 8  # Number of trail segments
PACKET_TRAIL_ALPHA = 0.3  # Trail transparency
MAX_PACKETS_ON_SCREEN = 50  # Performance limit
BASE_PACKET_SPEED = 0.08  # Base speed factor per frame (0-1 along path)
SLOW_DOWN_FACTOR = 1.2  # Speed scaling for longer paths


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AnimatedPacket:
    """Represents an animated packet moving along a route."""
    packet_id: int
    source: int
    destination: int
    current_path: List[int] = field(default_factory=list)
    progress: float = 0.0  # 0-1 along the path
    status: str = 'normal'  # 'normal', 'delivered', 'dropped', 'rerouted'
    creation_frame: int = 0
    trail_positions: List[Tuple[float, float]] = field(default_factory=list)
    speed: float = BASE_PACKET_SPEED
    visual_size: float = PACKET_SIZE_MIN
    
    def get_status_color(self) -> str:
        """Get color based on current packet status."""
        colors = {
            'normal': ColorPalette.PACKET_NORMAL,
            'delivered': ColorPalette.PACKET_DELIVERED,
            'dropped': ColorPalette.PACKET_DROPPED,
            'rerouted': ColorPalette.PACKET_REROUTED,
        }
        return colors.get(self.status, ColorPalette.PACKET_NORMAL)


# ============================================================================
# PACKET ANIMATOR COMPONENT
# ============================================================================

class PacketAnimator(VisualizationComponent):
    """
    Production-quality packet animation component for IoT network visualization.
    
    Animates multiple data packets moving smoothly along network routes with
    state-dependent coloring, variable speed, and motion trails. Efficiently
    handles 5-50 simultaneous packets while maintaining 30+ FPS performance.
    
    Integration with simulation:
    - Expects state dict with 'packets', 'active_routes', 'node_positions', 'sink_position'
    - Automatically synchronizes with simulation packet state
    - Caches path calculations for performance
    """
    
    def __init__(self, ax: plt.Axes, title: str = "Packet Animation"):
        """
        Initialize packet animator component.
        
        Args:
            ax: Matplotlib axes object
            title: Component title
        """
        super().__init__(ax, title)
        
        # Packet tracking
        self.animated_packets: Dict[int, AnimatedPacket] = {}
        self.packet_artists: Dict[int, mpatches.Patch] = {}
        self.trail_artists: Dict[int, List[Any]] = {}
        
        # Network state caching
        self.node_positions: Dict[int, Tuple[float, float]] = {}
        self.sink_position: Tuple[float, float] = (0, 0)
        self.path_cache: Dict[int, List[Tuple[float, float]]] = {}
        
        # State tracking
        self.last_frame = -1
        self.active_packet_ids: Set[int] = set()
    
    def initialize(self):
        """
        Initialize component - set up packet animation system.
        Called once before animation starts.
        """
        self.set_title(self.title)
        
        # Configure axes
        self.ax.set_aspect('equal')
        self.ax.set_facecolor(ColorPalette.BACKGROUND)
    
    def update(self, frame: int, state: Dict[str, Any]) -> List[Any]:
        """
        Update packet animation for given frame.
        
        Args:
            frame: Frame number
            state: Simulation state dictionary containing:
                - 'packets': List[Packet] (from iot_qlearning_simulation.py)
                - 'active_routes': Dict[int, List[int]] (current routes for each packet)
                - 'node_positions': Dict[int, Tuple[float, float]]
                - 'sink_position': Tuple[float, float]
        
        Returns:
            List of matplotlib artists for animation
        """
        artists = []
        
        # Extract state components
        packets = state.get('packets', [])
        active_routes = state.get('active_routes', {})
        self.node_positions = state.get('node_positions', {})
        self.sink_position = state.get('sink_position', (0, 0))
        
        # Update frame counter
        if frame != self.last_frame:
            self.last_frame = frame
        
        # Update or create animated packets
        self._create_new_packets(packets, active_routes)
        
        # Advance all packets along their paths
        self._advance_packet_positions()
        
        # Remove completed packets
        self._remove_completed_packets(packets)
        
        # Draw packet trails (motion blur effect)
        self._draw_packet_trails()
        
        # Collect all packet artists
        artists.extend(self.packet_artists.values())
        for trail_list in self.trail_artists.values():
            artists.extend(trail_list)
        
        return artists
    
    def _create_new_packets(self, packets: List[Any], 
                           active_routes: Dict[int, List[int]]):
        """
        Generate new packets from simulation state.
        
        Creates AnimatedPacket objects for new packets in the simulation,
        computing path coordinates and initial speed based on path length.
        Respects MAX_PACKETS_ON_SCREEN limit.
        
        Args:
            packets: List of Packet objects from simulation
            active_routes: Dict mapping packet_id to route (list of node_ids)
        """
        current_packet_ids = {p.packet_id for p in packets}
        
        # Process each packet from simulation
        for packet in packets:
            if packet.packet_id not in self.animated_packets:
                # New packet - add to tracking
                if len(self.animated_packets) < MAX_PACKETS_ON_SCREEN:
                    path = active_routes.get(packet.packet_id, 
                                            [packet.current_node, 0])
                    
                    animated = AnimatedPacket(
                        packet_id=packet.packet_id,
                        source=packet.source,
                        destination=packet.destination,
                        current_path=path,
                        progress=0.0,
                        status='normal',
                        creation_frame=self.last_frame,
                    )
                    
                    # Compute speed based on path length
                    path_length = len(path) - 1 if len(path) > 1 else 1
                    animated.speed = BASE_PACKET_SPEED / (1 + SLOW_DOWN_FACTOR * 
                                                          (path_length - 1) / 10.0)
                    
                    self.animated_packets[packet.packet_id] = animated
                    self.active_packet_ids.add(packet.packet_id)
            else:
                # Existing packet - update status
                animated = self.animated_packets[packet.packet_id]
                
                # Update route if changed
                new_route = active_routes.get(packet.packet_id, animated.current_path)
                if new_route != animated.current_path:
                    animated.current_path = new_route
                    animated.status = 'rerouted'
                    self.path_cache.pop(packet.packet_id, None)  # Invalidate cache
                
                # Update status based on packet state
                if packet.is_delivered:
                    animated.status = 'delivered'
                    animated.progress = 1.0  # Move to end
                elif packet.is_dropped:
                    animated.status = 'dropped'
                else:
                    if animated.status == 'normal':
                        pass  # Keep normal status
                    elif animated.status == 'rerouted' and not packet.is_dropped:
                        animated.status = 'normal'  # Transition back to normal
    
    def _advance_packet_positions(self):
        """
        Move each packet one step along its path.
        
        Updates progress for each animated packet based on its speed,
        respecting the 0-1 progress range. Handles packets nearing delivery.
        Also creates/updates visual packet artists.
        """
        for packet_id, animated in self.animated_packets.items():
            if animated.status != 'delivered' and animated.status != 'dropped':
                # Advance progress
                animated.progress = min(1.0, animated.progress + animated.speed)
            elif animated.status == 'delivered' or animated.status == 'dropped':
                # Ensure completed packets stay at end
                animated.progress = 1.0
            
            # Update or create packet artist
            self._update_packet_artist(animated)
    
    def _remove_completed_packets(self, packets: List[Any]):
        """
        Clean up delivered/dropped packets after visual persistence.
        
        Removes packet artists and data structures for packets that have
        reached their final state and had time to display completion.
        
        Args:
            packets: List of Packet objects from current simulation state
        """
        current_packet_ids = {p.packet_id for p in packets}
        packets_to_remove = []
        
        for packet_id, animated in self.animated_packets.items():
            # Remove if packet finished (delivered/dropped) and progress is complete
            is_finished = (animated.status in ['delivered', 'dropped'] and 
                          animated.progress >= 1.0)
            is_gone_from_sim = packet_id not in current_packet_ids
            
            if is_finished and is_gone_from_sim:
                packets_to_remove.append(packet_id)
        
        # Clean up removed packets
        for packet_id in packets_to_remove:
            # Remove artists
            if packet_id in self.packet_artists:
                self.packet_artists[packet_id].remove()
                del self.packet_artists[packet_id]
            
            if packet_id in self.trail_artists:
                for trail_art in self.trail_artists[packet_id]:
                    trail_art.remove()
                del self.trail_artists[packet_id]
            
            # Remove from tracking
            del self.animated_packets[packet_id]
            self.active_packet_ids.discard(packet_id)
            self.path_cache.pop(packet_id, None)
    
    def _draw_packet_trails(self):
        """
        Add motion trails (optional, for visual appeal).
        
        Creates fading trail segments behind each packet to show motion blur
        and direction. Trail becomes shorter and more transparent as it fades.
        """
        for packet_id, animated in self.animated_packets.items():
            # Get current position
            current_pos = self._get_packet_position(animated)
            if current_pos is None:
                continue
            
            # Update trail
            if len(animated.trail_positions) == 0:
                animated.trail_positions.append(current_pos)
            elif current_pos != animated.trail_positions[-1]:
                animated.trail_positions.append(current_pos)
                # Keep trail length bounded
                if len(animated.trail_positions) > PACKET_TRAIL_LENGTH:
                    animated.trail_positions.pop(0)
            
            # Redraw trail artists
            if packet_id in self.trail_artists:
                for art in self.trail_artists[packet_id]:
                    art.remove()
            
            self.trail_artists[packet_id] = []
            
            # Draw trail segments with decreasing opacity
            if len(animated.trail_positions) > 1:
                for i, (x, y) in enumerate(animated.trail_positions[:-1]):
                    x_next, y_next = animated.trail_positions[i + 1]
                    
                    # Fade based on position in trail
                    fade = (i + 1) / len(animated.trail_positions)
                    trail_alpha = PACKET_TRAIL_ALPHA * fade * 0.7
                    trail_size = animated.visual_size * fade * 0.6
                    
                    trail_circle = Circle(
                        (x, y), trail_size / 200.0,
                        color=animated.get_status_color(),
                        alpha=trail_alpha,
                        zorder=8
                    )
                    self.ax.add_patch(trail_circle)
                    self.trail_artists[packet_id].append(trail_circle)
                    self.artists.append(trail_circle)
    
    def _update_packet_artist(self, animated: AnimatedPacket):
        """
        Update or create visual packet artist for animated packet.
        
        Creates a new packet circle artist if one doesn't exist, or updates
        the position and color of existing artist. Handles packet lifecycle
        and visual state transitions.
        
        Args:
            animated: AnimatedPacket to visualize
        """
        # Get packet position
        current_pos = self._get_packet_position(animated)
        if current_pos is None:
            return
        
        x, y = current_pos
        
        # Calculate visual size based on state
        if animated.status == 'delivered':
            animated.visual_size = PACKET_SIZE_MAX
        elif animated.status == 'dropped':
            animated.visual_size = PACKET_SIZE_MIN * 0.8
        else:
            # Grow slightly for normal and rerouted packets
            animated.visual_size = PACKET_SIZE_MIN + (
                (PACKET_SIZE_MAX - PACKET_SIZE_MIN) * 0.3
            )
        
        # Create or update artist
        if animated.packet_id not in self.packet_artists:
            # Create new packet artist
            radius = animated.visual_size / 200.0
            packet_circle = Circle(
                (x, y), radius,
                color=animated.get_status_color(),
                alpha=0.85,
                zorder=10,
                edgecolor='white' if animated.status in ['delivered', 'rerouted'] else 'none',
                linewidth=1.0 if animated.status in ['delivered', 'rerouted'] else 0
            )
            self.ax.add_patch(packet_circle)
            self.packet_artists[animated.packet_id] = packet_circle
            self.artists.append(packet_circle)
        else:
            # Update existing artist
            artist = self.packet_artists[animated.packet_id]
            artist.center = (x, y)
            artist.set_color(animated.get_status_color())
            
            # Update size
            new_radius = animated.visual_size / 200.0
            artist.set_radius(new_radius)
            
            # Update edge based on state
            if animated.status in ['delivered', 'rerouted']:
                artist.set_edgecolor('white')
                artist.set_linewidth(1.0)
            else:
                artist.set_edgecolor('none')
                artist.set_linewidth(0)
    
    def _get_packet_position(self, animated: AnimatedPacket) -> Optional[Tuple[float, float]]:
        """
        Calculate packet position along its path.
        
        Interpolates position between path nodes based on progress (0-1).
        Uses cached path coordinates for efficiency.
        
        Args:
            animated: AnimatedPacket object
        
        Returns:
            Tuple of (x, y) coordinates or None if path is invalid
        """
        if not animated.current_path or len(animated.current_path) < 1:
            return None
        
        packet_id = animated.packet_id
        
        # Use or compute cached path
        if packet_id not in self.path_cache:
            path_coords = []
            for node_id in animated.current_path:
                if node_id in self.node_positions:
                    path_coords.append(self.node_positions[node_id])
                elif node_id == 0:  # Sink node
                    path_coords.append(self.sink_position)
            
            if len(path_coords) != len(animated.current_path):
                return None  # Path has invalid nodes
            
            self.path_cache[packet_id] = path_coords
        
        path_coords = self.path_cache[packet_id]
        if not path_coords or len(path_coords) < 1:
            return None
        
        if len(path_coords) == 1:
            return path_coords[0]
        
        # Interpolate position along path
        progress = animated.progress
        segment_index = int(progress * (len(path_coords) - 1))
        segment_index = min(segment_index, len(path_coords) - 2)
        
        # Local progress within segment
        segment_progress = (progress * (len(path_coords) - 1)) - segment_index
        segment_progress = smooth_step(segment_progress)  # Smooth interpolation
        
        x1, y1 = path_coords[segment_index]
        x2, y2 = path_coords[segment_index + 1]
        
        x = lerp(x1, x2, segment_progress)
        y = lerp(y1, y2, segment_progress)
        
        return (x, y)
    
    def _update_packet_visuals(self):
        """
        Update packet visual representation (size and color).
        
        Adjusts packet size based on state and progress for better visibility
        of completed packets. Updates all packet artists.
        """
        for packet_id, animated in self.animated_packets.items():
            # Calculate visual size based on state
            if animated.status == 'delivered':
                animated.visual_size = PACKET_SIZE_MAX
            elif animated.status == 'dropped':
                animated.visual_size = PACKET_SIZE_MIN * 0.8
            else:
                # Grow slightly for normal and rerouted packets
                animated.visual_size = PACKET_SIZE_MIN + (
                    (PACKET_SIZE_MAX - PACKET_SIZE_MIN) * 0.3
                )
            
            # Update artist if it exists
            if packet_id in self.packet_artists:
                artist = self.packet_artists[packet_id]
                artist.set_color(animated.get_status_color())
                
                # Update size
                new_radius = animated.visual_size / 200.0
                artist.set_radius(new_radius)
    
    def clear(self):
        """Clear all packet artists and reset state."""
        # Remove all artists
        for artist in self.packet_artists.values():
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        
        for trail_list in self.trail_artists.values():
            for art in trail_list:
                try:
                    art.remove()
                except (ValueError, AttributeError):
                    pass
        
        # Reset state
        self.packet_artists.clear()
        self.trail_artists.clear()
        self.animated_packets.clear()
        self.active_packet_ids.clear()
        self.path_cache.clear()
        
        # Clear collections list to prevent removal errors
        self.collections.clear()
        self.artists.clear()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_packet_animator(ax: plt.Axes, title: str = "Packet Animation") -> PacketAnimator:
    """
    Factory function to create and initialize a packet animator.
    
    Args:
        ax: Matplotlib axes object
        title: Component title
    
    Returns:
        Initialized PacketAnimator instance
    """
    animator = PacketAnimator(ax, title)
    animator.initialize()
    return animator
