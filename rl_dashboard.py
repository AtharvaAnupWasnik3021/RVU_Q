"""
Real-time Reinforcement Learning Metrics Dashboard.

Displays live RL metrics for IoT network optimization with IEEE-conference-quality
visualization. Supports real-time updates at 30+ FPS with color-coded metrics,
sparklines, and trend visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.text import Text
from typing import Dict, List, Any, Optional
from collections import deque

from publication_viz_core import (
    VisualizationComponent,
    ColorPalette,
    Typography,
)


class RLDashboard(VisualizationComponent):
    """
    Professional RL metrics dashboard for IoT network visualization.
    
    Displays real-time reinforcement learning metrics in a compact panel,
    including scalar values, trends, and sparklines. Optimized for 30+ FPS
    performance with efficient text updates and minimal redraws.
    """
    
    # Configuration constants
    MAX_HISTORY = 30  # Keep last 30 frames for sparklines
    METRIC_ROW_HEIGHT = 0.12
    COLUMN_WIDTHS = [0.40, 0.30, 0.30]  # Left, middle, right
    
    def __init__(self, ax: plt.Axes, title: str = "RL Metrics"):
        """
        Initialize RL Dashboard component.
        
        Args:
            ax: Matplotlib axes object
            title: Component title (default: "RL Metrics")
        """
        super().__init__(ax, title)
        
        # Text elements for scalar metrics
        self.text_metrics = {}
        self.text_labels = {}
        
        # Line artists for sparklines
        self.lines_sparkline = {}
        
        # Bar chart artists
        self.bar_exploitation = None
        self.bar_success_rate = None
        
        # Episode counter text
        self.text_episode = None
        
        # History buffers (deque for efficient operations)
        self.reward_history = deque(maxlen=self.MAX_HISTORY)
        self.q_mean_history = deque(maxlen=self.MAX_HISTORY)
        self.epsilon_history = deque(maxlen=self.MAX_HISTORY)
        
        # Previous values for efficient updates
        self.prev_values = {}
        
        # Configure axis
        self._setup_axis()
    
    def _setup_axis(self):
        """Configure axis styling for dashboard."""
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.ax.set_aspect('equal')
    
    def initialize(self):
        """
        Initialize metric displays, text boxes, and mini-plots.
        
        Sets up layout with three columns:
        - Left (40%): Scalar metric values
        - Middle (30%): Sparkline charts
        - Right (30%): Bar charts and counters
        """
        self.ax.clear()
        self._setup_axis()
        
        # Title
        title_text = self.ax.text(
            0.5, 0.95,
            self.title,
            fontsize=Typography.SIZE_SECTION,
            fontweight=Typography.WEIGHT_BOLD,
            color=ColorPalette.TEXT_PRIMARY,
            ha='center',
            va='top',
            transform=self.ax.transAxes
        )
        self.artists.append(title_text)
        
        # Define metrics layout (metric_name, y_position, unit)
        metrics_config = [
            ('Current Reward', 0.82, ''),
            ('Avg Reward', 0.68, ''),
            ('Q-Value Mean', 0.54, ''),
            ('Q-Value Var', 0.40, ''),
            ('Explore Rate ε', 0.26, ''),
            ('Route Cost', 0.12, 'hops'),
        ]
        
        # Left column: Labels and scalar values
        for metric_name, y_pos, unit in metrics_config:
            # Label
            label = self.ax.text(
                0.02, y_pos,
                f"{metric_name}:",
                fontsize=Typography.SIZE_SMALL,
                color=ColorPalette.TEXT_SECONDARY,
                ha='left',
                va='center',
                transform=self.ax.transAxes
            )
            self.artists.append(label)
            self.text_labels[metric_name] = label
            
            # Value (placeholder)
            value = self.ax.text(
                0.22, y_pos,
                '0.000',
                fontsize=Typography.SIZE_LABEL,
                color=ColorPalette.TEXT_PRIMARY,
                ha='left',
                va='center',
                fontweight=Typography.WEIGHT_BOLD,
                transform=self.ax.transAxes
            )
            self.artists.append(value)
            self.text_metrics[metric_name] = value
        
        # Middle column: Sparklines (small line charts)
        sparkline_metrics = [
            ('reward_spark', 0.82, 'Reward Trend'),
            ('q_mean_spark', 0.54, 'Q-Mean Trend'),
            ('epsilon_spark', 0.26, 'ε Decay'),
        ]
        
        for metric_key, y_pos, label_text in sparkline_metrics:
            # Sparkline background
            sparkline_bg = mpatches.FancyBboxPatch(
                (0.35, y_pos - 0.05),
                0.22, 0.08,
                boxstyle="round,pad=0.002",
                transform=self.ax.transAxes,
                facecolor=ColorPalette.GRID_COLOR,
                edgecolor=ColorPalette.TEXT_SECONDARY,
                linewidth=0.5,
                alpha=0.3
            )
            self.ax.add_patch(sparkline_bg)
            self.artists.append(sparkline_bg)
            
            # Sparkline axes (relative coordinates within the box)
            line, = self.ax.plot([], [], color=ColorPalette.PROTOCOL_QLEARNING,
                                linewidth=1.5, alpha=0.8, transform=self.ax.transAxes)
            self.lines_sparkline[metric_key] = line
            self.artists.append(line)
        
        # Right column: Exploitation % and Success Rate %
        # Exploitation bar
        exploit_bg = mpatches.FancyBboxPatch(
            (0.65, 0.68 - 0.05),
            0.32, 0.08,
            boxstyle="round,pad=0.002",
            transform=self.ax.transAxes,
            facecolor=ColorPalette.GRID_COLOR,
            edgecolor=ColorPalette.TEXT_SECONDARY,
            linewidth=0.5,
            alpha=0.3
        )
        self.ax.add_patch(exploit_bg)
        self.artists.append(exploit_bg)
        
        # Exploitation bar (filled rectangle)
        self.bar_exploitation = mpatches.Rectangle(
            (0.66, 0.68 - 0.04),
            0.0, 0.06,
            transform=self.ax.transAxes,
            facecolor=ColorPalette.PROTOCOL_QLEARNING,
            edgecolor='none',
            alpha=0.7
        )
        self.ax.add_patch(self.bar_exploitation)
        self.artists.append(self.bar_exploitation)
        
        # Exploitation label
        exploit_label = self.ax.text(
            0.65, 0.77,
            'Exploit %:',
            fontsize=Typography.SIZE_SMALL,
            color=ColorPalette.TEXT_SECONDARY,
            ha='left',
            va='top',
            transform=self.ax.transAxes
        )
        self.artists.append(exploit_label)
        
        # Exploitation value
        exploit_val = self.ax.text(
            0.98, 0.71,
            '0%',
            fontsize=Typography.SIZE_SMALL,
            color=ColorPalette.TEXT_PRIMARY,
            ha='right',
            va='center',
            transform=self.ax.transAxes
        )
        self.artists.append(exploit_val)
        self.text_metrics['Exploitation %'] = exploit_val
        
        # Success Rate bar
        success_bg = mpatches.FancyBboxPatch(
            (0.65, 0.40 - 0.05),
            0.32, 0.08,
            boxstyle="round,pad=0.002",
            transform=self.ax.transAxes,
            facecolor=ColorPalette.GRID_COLOR,
            edgecolor=ColorPalette.TEXT_SECONDARY,
            linewidth=0.5,
            alpha=0.3
        )
        self.ax.add_patch(success_bg)
        self.artists.append(success_bg)
        
        # Success rate bar (filled rectangle)
        self.bar_success_rate = mpatches.Rectangle(
            (0.66, 0.40 - 0.04),
            0.0, 0.06,
            transform=self.ax.transAxes,
            facecolor=ColorPalette.HEALTHY_NODE,
            edgecolor='none',
            alpha=0.7
        )
        self.ax.add_patch(self.bar_success_rate)
        self.artists.append(self.bar_success_rate)
        
        # Success rate label
        success_label = self.ax.text(
            0.65, 0.49,
            'Success %:',
            fontsize=Typography.SIZE_SMALL,
            color=ColorPalette.TEXT_SECONDARY,
            ha='left',
            va='top',
            transform=self.ax.transAxes
        )
        self.artists.append(success_label)
        
        # Success rate value
        success_val = self.ax.text(
            0.98, 0.43,
            '0%',
            fontsize=Typography.SIZE_SMALL,
            color=ColorPalette.TEXT_PRIMARY,
            ha='right',
            va='center',
            transform=self.ax.transAxes
        )
        self.artists.append(success_val)
        self.text_metrics['Packet Success Rate'] = success_val
        
        # Episode counter (bottom-right)
        self.text_episode = self.ax.text(
            0.98, 0.08,
            'Ep: 0',
            fontsize=Typography.SIZE_LABEL,
            color=ColorPalette.PROTOCOL_QLEARNING,
            ha='right',
            va='bottom',
            fontweight=Typography.WEIGHT_BOLD,
            transform=self.ax.transAxes
        )
        self.artists.append(self.text_episode)
        
        # Initialize history buffers
        self.reward_history.clear()
        self.q_mean_history.clear()
        self.epsilon_history.clear()
    
    def update(self, frame: int, state: Dict[str, Any]) -> List[Any]:
        """
        Update all metrics and return artist list.
        
        Args:
            frame: Frame number
            state: Simulation state dictionary with keys:
                - current_reward: float
                - average_reward: float
                - reward_history: List[float] (last 30 frames)
                - q_value_mean: float
                - q_value_variance: float
                - epsilon: float (exploration rate, 0-1)
                - exploitation_percentage: float (0-100)
                - current_route_cost: int (hops)
                - packet_success_rate: float (0-100)
                - episode: int (episode counter)
                - q_values_history: List[float] (for trends)
        
        Returns:
            List of matplotlib artists to update
        """
        updated_artists = []
        
        # Update scalar metrics
        updated_artists.extend(self._update_scalar_metrics(state))
        
        # Update sparklines
        updated_artists.extend(self._update_sparklines(state))
        
        # Update bar charts
        updated_artists.extend(self._update_bars(state))
        
        # Update episode counter
        updated_artists.append(self._update_episode_counter(state))
        
        return updated_artists
    
    def _update_scalar_metrics(self, state: Dict[str, Any]) -> List[Any]:
        """
        Update scalar metric displays with color coding.
        
        Returns:
            List of updated text artists
        """
        updated = []
        
        # Current Reward (color-coded)
        current_reward = state.get('current_reward', 0.0)
        if self.text_metrics['Current Reward'].get_text() != self._format_value(current_reward):
            reward_color = ColorPalette.reward_color(current_reward)
            self.text_metrics['Current Reward'].set_color(reward_color)
            self.text_metrics['Current Reward'].set_text(self._format_value(current_reward))
            updated.append(self.text_metrics['Current Reward'])
        
        # Average Reward
        avg_reward = state.get('average_reward', 0.0)
        if self.text_metrics['Avg Reward'].get_text() != self._format_value(avg_reward):
            self.text_metrics['Avg Reward'].set_text(self._format_value(avg_reward))
            updated.append(self.text_metrics['Avg Reward'])
        
        # Q-Value Mean
        q_mean = state.get('q_value_mean', 0.0)
        if self.text_metrics['Q-Value Mean'].get_text() != self._format_value(q_mean):
            self.text_metrics['Q-Value Mean'].set_text(self._format_value(q_mean))
            updated.append(self.text_metrics['Q-Value Mean'])
        
        # Q-Value Variance
        q_var = state.get('q_value_variance', 0.0)
        if self.text_metrics['Q-Value Var'].get_text() != self._format_value(q_var):
            self.text_metrics['Q-Value Var'].set_text(self._format_value(q_var))
            updated.append(self.text_metrics['Q-Value Var'])
        
        # Exploration Rate ε
        epsilon = state.get('epsilon', 1.0)
        if self.text_metrics['Explore Rate ε'].get_text() != f"{epsilon:.3f}":
            self.text_metrics['Explore Rate ε'].set_text(f"{epsilon:.3f}")
            updated.append(self.text_metrics['Explore Rate ε'])
        
        # Route Cost (in hops)
        route_cost = state.get('current_route_cost', 0)
        if self.text_metrics['Route Cost'].get_text() != f"{route_cost}":
            self.text_metrics['Route Cost'].set_text(f"{route_cost}")
            updated.append(self.text_metrics['Route Cost'])
        
        return updated
    
    def _update_sparklines(self, state: Dict[str, Any]) -> List[Any]:
        """
        Update mini line charts with trend data.
        
        Returns:
            List of updated line artists
        """
        updated = []
        
        # Reward history sparkline
        reward_hist = state.get('reward_history', [])
        if reward_hist:
            self.reward_history.extend(reward_hist[-1:])  # Add latest
            
            if len(self.reward_history) > 1:
                x_data = np.linspace(0.35, 0.57, len(self.reward_history))
                y_data = np.interp(
                    np.linspace(0, 1, len(self.reward_history)),
                    np.linspace(0, 1, len(self.reward_history)),
                    np.array(list(self.reward_history))
                )
                # Normalize to plot area (0.82 ± 0.04)
                y_min = min(list(self.reward_history) + [0])
                y_max = max(list(self.reward_history) + [1])
                if y_max > y_min:
                    y_norm = 0.78 + (y_data - y_min) / (y_max - y_min) * 0.08
                else:
                    y_norm = 0.82
                
                self.lines_sparkline['reward_spark'].set_data(x_data, y_norm)
                updated.append(self.lines_sparkline['reward_spark'])
        
        # Q-mean history sparkline
        q_hist = state.get('q_values_history', [])
        if q_hist:
            self.q_mean_history.extend(q_hist[-1:])
            
            if len(self.q_mean_history) > 1:
                x_data = np.linspace(0.35, 0.57, len(self.q_mean_history))
                y_data = np.array(list(self.q_mean_history))
                # Normalize to plot area (0.54 ± 0.04)
                y_min = np.min(y_data)
                y_max = np.max(y_data)
                if y_max > y_min:
                    y_norm = 0.50 + (y_data - y_min) / (y_max - y_min) * 0.08
                else:
                    y_norm = np.full_like(y_data, 0.54)
                
                self.lines_sparkline['q_mean_spark'].set_data(x_data, y_norm)
                updated.append(self.lines_sparkline['q_mean_spark'])
        
        # Epsilon decay sparkline
        epsilon = state.get('epsilon', 1.0)
        self.epsilon_history.append(epsilon)
        
        if len(self.epsilon_history) > 1:
            x_data = np.linspace(0.35, 0.57, len(self.epsilon_history))
            y_data = np.array(list(self.epsilon_history))
            # Normalize to plot area (0.26 ± 0.04, inverted since epsilon decreases)
            y_norm = 0.22 + (1.0 - y_data) * 0.08
            
            self.lines_sparkline['epsilon_spark'].set_data(x_data, y_norm)
            updated.append(self.lines_sparkline['epsilon_spark'])
        
        return updated
    
    def _update_bars(self, state: Dict[str, Any]) -> List[Any]:
        """
        Update bar charts (exploitation % and success rate %).
        
        Returns:
            List of updated bar artists
        """
        updated = []
        
        # Exploitation bar (0-100%)
        exploit_pct = state.get('exploitation_percentage', 0.0)
        exploit_pct = max(0, min(100, exploit_pct))  # Clamp to [0, 100]
        
        width = 0.30 * (exploit_pct / 100.0)  # Scale to max width
        self.bar_exploitation.set_width(width)
        updated.append(self.bar_exploitation)
        
        # Exploitation value text
        self.text_metrics['Exploitation %'].set_text(f"{exploit_pct:.0f}%")
        updated.append(self.text_metrics['Exploitation %'])
        
        # Success rate bar (0-100%)
        success_pct = state.get('packet_success_rate', 0.0)
        success_pct = max(0, min(100, success_pct))  # Clamp to [0, 100]
        
        width = 0.30 * (success_pct / 100.0)  # Scale to max width
        self.bar_success_rate.set_width(width)
        updated.append(self.bar_success_rate)
        
        # Success rate value text
        self.text_metrics['Packet Success Rate'].set_text(f"{success_pct:.0f}%")
        updated.append(self.text_metrics['Packet Success Rate'])
        
        return updated
    
    def _update_episode_counter(self, state: Dict[str, Any]) -> Text:
        """
        Update episode counter display.
        
        Args:
            state: Simulation state
        
        Returns:
            Updated episode counter text artist
        """
        episode = state.get('episode', 0)
        self.text_episode.set_text(f"Ep: {episode}")
        return self.text_episode
    
    def _format_value(self, value: float, precision: int = 3) -> str:
        """
        Format metric value with appropriate precision.
        
        Args:
            value: Numeric value to format
            precision: Decimal places (default: 3)
        
        Returns:
            Formatted string representation
        """
        if abs(value) < 0.001:
            return f"{value:.2e}"
        elif abs(value) < 1:
            return f"{value:.{precision}f}"
        elif abs(value) < 100:
            return f"{value:.{max(1, precision-1)}f}"
        else:
            return f"{value:.1f}"
