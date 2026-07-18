"""
Security-aware routing visualization component.

Displays real-time routing decisions with security metrics, highlighting
unsafe vs. safe paths, and visualizing threat detection and rerouting events.
Production-ready component for IEEE-quality IoT security visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import deque
import time

from publication_viz_core import (
    VisualizationComponent,
    ColorPalette,
    Typography,
)


@dataclass
class Annotation:
    """Text annotation with fade-out functionality."""
    text: str
    x: float
    y: float
    created_at: float
    fade_duration: float = 2.5
    alpha: float = 1.0
    max_alpha: float = 0.95
    
    def get_alpha(self, current_time: float) -> float:
        """Calculate alpha based on fade duration."""
        elapsed = current_time - self.created_at
        if elapsed > self.fade_duration:
            return 0.0
        # Fade out during last 0.5 seconds
        fade_start = self.fade_duration - 0.5
        if elapsed < fade_start:
            return self.max_alpha
        fade_progress = (elapsed - fade_start) / 0.5
        return self.max_alpha * (1.0 - fade_progress)
    
    def is_expired(self, current_time: float) -> bool:
        """Check if annotation has faded out completely."""
        return current_time - self.created_at > self.fade_duration


@dataclass
class RoutingState:
    """Tracks routing state for visualization."""
    active_route: List[int] = field(default_factory=list)
    old_route: List[int] = field(default_factory=list)
    candidate_routes: List[List[int]] = field(default_factory=list)
    infected_nodes: Set[int] = field(default_factory=set)
    newly_infected: Set[int] = field(default_factory=set)
    route_changed: bool = False
    safety_confidence: float = 1.0
    reroute_reason: str = ""
    
    def update_from_state(self, state: Dict[str, Any]):
        """Update internal state from simulation state dict."""
        self.active_route = state.get('active_route', self.active_route)
        self.infected_nodes = state.get('infected_nodes', self.infected_nodes)
        self.newly_infected = state.get('newly_infected', self.newly_infected)
        self.route_changed = state.get('route_change', False)
        self.old_route = state.get('old_route', self.old_route)
        self.safety_confidence = state.get('safety_confidence', 1.0)
        self.reroute_reason = state.get('reroute_reason', "")


class SecurityRoutingDisplay(VisualizationComponent):
    """
    Real-time security-aware routing decision visualization.
    
    Shows routing paths with security highlighting, threat detection,
    and rerouting animations. Integrates network topology visualization
    with security metrics and routing state.
    """
    
    def __init__(self, ax: plt.Axes, title: str = "Security-Aware Routing",
                 network_graph: Optional[Dict[int, List[Tuple[float, float]]]] = None):
        """
        Initialize security routing display.
        
        Args:
            ax: Matplotlib axes for rendering
            title: Component title
            network_graph: Dict mapping node IDs to (x, y) positions
        """
        super().__init__(ax, title)
        
        # Network topology
        self.network_graph = network_graph or {}
        
        # Routing state tracking
        self.routing_state = RoutingState()
        self.previous_state = RoutingState()
        
        # Annotation management
        self.annotations: deque = deque(maxlen=5)  # Max 5 annotations
        self.annotation_artists = []
        
        # Visual elements
        self.active_route_lines = []
        self.unsafe_route_lines = []
        self.safe_route_lines = []
        self.threat_indicators = []
        
        # Timing
        self.frame_times = {}
        self.threat_animation_time = 0
        self.current_time = time.time()
        
        # Configuration
        self.unsafe_alpha = 0.5
        self.safe_alpha = 0.6
        self.line_width = 3.0
        self.highlight_width = 5.0
        
    def initialize(self):
        """Initialize display layers and styling."""
        self.ax.set_title(
            self.title,
            fontsize=Typography.SIZE_SECTION,
            fontweight=Typography.WEIGHT_BOLD,
            color=ColorPalette.TEXT_PRIMARY,
            pad=10,
        )
        
        # Set up axis
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.1, color=ColorPalette.GRID_COLOR)
        self.ax.set_facecolor(ColorPalette.BACKGROUND)
        
        # Initialize state tracking
        self.previous_state = RoutingState()
        self.routing_state = RoutingState()
        self.annotations.clear()
        self.current_time = time.time()
    
    def update(self, frame: int, state: Dict[str, Any]) -> List[Any]:
        """
        Update routing visualization for current frame.
        
        Args:
            frame: Current frame number
            state: Simulation state dict with routing information
        
        Returns:
            List of matplotlib artists for animation
        """
        updated_artists = []
        self.current_time = time.time()
        
        # Update routing state from simulation
        self.previous_state = RoutingState(
            active_route=self.routing_state.active_route.copy(),
            old_route=self.routing_state.old_route.copy(),
            infected_nodes=self.routing_state.infected_nodes.copy(),
        )
        self.routing_state.update_from_state(state)
        
        # Clear previous routing visualizations
        for line in self.active_route_lines + self.unsafe_route_lines + self.safe_route_lines:
            if hasattr(line, 'remove'):
                try:
                    line.remove()
                except (ValueError, AttributeError):
                    pass
        
        self.active_route_lines.clear()
        self.unsafe_route_lines.clear()
        self.safe_route_lines.clear()
        
        # Detect threats and rerouting events
        threat_detected = self._detect_threat_and_reroute()
        
        # Visualize routing decisions
        if self.routing_state.route_changed and self.previous_state.active_route:
            # Show old route turning unsafe
            unsafe_artists = self._highlight_unsafe_routes()
            self.unsafe_route_lines.extend(unsafe_artists)
            updated_artists.extend(unsafe_artists)
        
        # Highlight safe (active) route
        if self.routing_state.active_route:
            safe_artists = self._highlight_safe_routes()
            self.safe_route_lines.extend(safe_artists)
            updated_artists.extend(safe_artists)
        
        # Add annotations for routing events
        annotation_artists = self._add_annotations(threat_detected)
        updated_artists.extend(annotation_artists)
        
        # Fade out and remove expired annotations
        cleaned_artists = self._fade_annotation_text()
        updated_artists.extend(cleaned_artists)
        
        # Add annotations to artists list for animation
        self.artists.extend(updated_artists)
        
        return updated_artists
    
    def _highlight_unsafe_routes(self) -> List[Any]:
        """
        Draw highlighting for unsafe/previous routes.
        
        Returns:
            List of matplotlib artists
        """
        artists = []
        
        if not self.previous_state.active_route or len(self.network_graph) < 2:
            return artists
        
        # Extract edge segments from previous route
        edges = []
        edge_segments = []
        
        for i in range(len(self.previous_state.active_route) - 1):
            node1 = self.previous_state.active_route[i]
            node2 = self.previous_state.active_route[i + 1]
            
            if node1 in self.network_graph and node2 in self.network_graph:
                pos1 = self.network_graph[node1]
                pos2 = self.network_graph[node2]
                edge_segments.append([pos1, pos2])
                edges.append((node1, node2))
        
        if edge_segments:
            # Create line collection for old route
            lc = LineCollection(
                edge_segments,
                linewidths=self.highlight_width,
                colors=ColorPalette.FAILED_EDGE,
                alpha=self.unsafe_alpha,
                zorder=1,
            )
            self.ax.add_collection(lc)
            artists.append(lc)
            self.collections.append(lc)
        
        return artists
    
    def _highlight_safe_routes(self) -> List[Any]:
        """
        Draw highlighting for safe/active routes.
        
        Returns:
            List of matplotlib artists
        """
        artists = []
        
        if not self.routing_state.active_route or len(self.network_graph) < 2:
            return artists
        
        # Extract edge segments from active route
        edge_segments = []
        
        for i in range(len(self.routing_state.active_route) - 1):
            node1 = self.routing_state.active_route[i]
            node2 = self.routing_state.active_route[i + 1]
            
            if node1 in self.network_graph and node2 in self.network_graph:
                pos1 = self.network_graph[node1]
                pos2 = self.network_graph[node2]
                edge_segments.append([pos1, pos2])
        
        if edge_segments:
            # Determine color based on confidence
            if self.routing_state.safety_confidence > 0.8:
                color = ColorPalette.ACTIVE_EDGE  # Cyan for high confidence
            else:
                color = ColorPalette.QLEARN_EDGE  # Green for lower confidence
            
            # Create line collection for active route
            lc = LineCollection(
                edge_segments,
                linewidths=self.highlight_width,
                colors=color,
                alpha=self.safe_alpha,
                zorder=2,
            )
            self.ax.add_collection(lc)
            artists.append(lc)
            self.collections.append(lc)
        
        return artists
    
    def _detect_threat_and_reroute(self) -> bool:
        """
        Track when rerouting occurs and detect threat patterns.
        
        Returns:
            True if threat was detected in this frame
        """
        threat_detected = False
        
        # Check for new infections
        new_threats = self.routing_state.newly_infected - self.previous_state.newly_infected
        if new_threats:
            threat_detected = True
        
        # Check for route changes
        route_changed = (
            self.routing_state.active_route != self.previous_state.active_route
            and self.routing_state.route_changed
        )
        
        return threat_detected or route_changed
    
    def _add_annotations(self, threat_detected: bool) -> List[Any]:
        """
        Add text annotations for routing events.
        
        Args:
            threat_detected: Whether a threat was detected
        
        Returns:
            List of matplotlib text artists
        """
        artists = []
        
        # Detect newly infected nodes for annotation positioning
        newly_infected = self.routing_state.newly_infected
        
        if threat_detected and newly_infected and self.network_graph:
            # Get position of first newly infected node
            infected_node = list(newly_infected)[0]
            if infected_node in self.network_graph:
                x, y = self.network_graph[infected_node]
                
                # Add "Threat detected" annotation
                self.annotations.append(
                    Annotation(
                        text="⚠ Threat detected",
                        x=x,
                        y=y + 0.3,
                        created_at=self.current_time,
                    )
                )
        
        # Annotation for route change
        if self.routing_state.route_changed and self.routing_state.active_route:
            if len(self.routing_state.active_route) > 0:
                # Position near start of new route
                start_node = self.routing_state.active_route[0]
                if start_node in self.network_graph:
                    x, y = self.network_graph[start_node]
                    
                    self.annotations.append(
                        Annotation(
                            text="↻ Route recomputed",
                            x=x - 0.3,
                            y=y - 0.3,
                            created_at=self.current_time,
                        )
                    )
            
            # Position near end of new route
            if len(self.routing_state.active_route) > 1:
                end_node = self.routing_state.active_route[-1]
                if end_node in self.network_graph:
                    x, y = self.network_graph[end_node]
                    
                    if self.routing_state.safety_confidence > 0.8:
                        text = "✓ Secure path selected"
                    else:
                        text = "◇ Path selected (risk)"
                    
                    self.annotations.append(
                        Annotation(
                            text=text,
                            x=x,
                            y=y - 0.3,
                            created_at=self.current_time,
                        )
                    )
        
        # Render all annotations
        for annotation in self.annotations:
            alpha = annotation.get_alpha(self.current_time)
            if alpha > 0:
                text_artist = self.ax.text(
                    annotation.x,
                    annotation.y,
                    annotation.text,
                    fontsize=Typography.SIZE_LABEL,
                    color=ColorPalette.TEXT_PRIMARY,
                    alpha=alpha,
                    ha='center',
                    va='center',
                    fontweight=Typography.WEIGHT_BOLD,
                    bbox=dict(
                        boxstyle='round,pad=0.4',
                        facecolor=ColorPalette.BACKGROUND,
                        edgecolor=ColorPalette.TEXT_PRIMARY,
                        alpha=alpha * 0.7,
                        linewidth=1.0,
                    ),
                    zorder=10,
                )
                artists.append(text_artist)
                self.annotation_artists.append(text_artist)
        
        return artists
    
    def _fade_annotation_text(self) -> List[Any]:
        """
        Fade out old annotations and clean up expired ones.
        
        Returns:
            List of matplotlib artists
        """
        artists = []
        
        # Remove expired annotations
        expired_count = 0
        for annotation in list(self.annotations):
            if annotation.is_expired(self.current_time):
                self.annotations.remove(annotation)
                expired_count += 1
        
        # Clean up expired text artists from axes
        for artist in list(self.annotation_artists):
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        
        self.annotation_artists.clear()
        
        return artists
    
    def get_routing_info(self) -> Dict[str, Any]:
        """
        Get current routing information for diagnostics.
        
        Returns:
            Dictionary with routing metrics
        """
        return {
            'active_route': self.routing_state.active_route.copy(),
            'route_hops': len(self.routing_state.active_route) - 1,
            'infected_nodes_count': len(self.routing_state.infected_nodes),
            'safety_confidence': self.routing_state.safety_confidence,
            'reroute_reason': self.routing_state.reroute_reason,
            'active_annotations': len(self.annotations),
        }
    
    def set_network_positions(self, positions: Dict[int, Tuple[float, float]]):
        """
        Set node positions for the network topology.
        
        Args:
            positions: Dict mapping node IDs to (x, y) coordinates
        """
        self.network_graph = positions.copy()
    
    def clear_routes(self):
        """Clear all route visualizations."""
        for line in self.active_route_lines + self.unsafe_route_lines + self.safe_route_lines:
            try:
                line.remove()
            except (ValueError, AttributeError):
                pass
        
        self.active_route_lines.clear()
        self.unsafe_route_lines.clear()
        self.safe_route_lines.clear()
    
    def clear(self):
        """Clear all artists and state."""
        self.clear_routes()
        
        # Clear annotations - don't add to artists tracking to avoid double removal
        for artist in self.annotation_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        
        self.annotation_artists.clear()
        self.annotations.clear()
        
        # Call parent clear only for non-annotation artists
        for artist in self.artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        
        self.artists.clear()
        
        for collection in self.collections:
            try:
                collection.remove()
            except (ValueError, AttributeError):
                pass
        
        self.collections.clear()
