"""
Performance metrics panel for real-time network visualization.

Displays 12 key network performance metrics in a professional 3x4 grid layout
with live updates, trend indicators, and confidence intervals suitable for
IEEE conference publications.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional, Any
from publication_viz_core import (
    VisualizationComponent, ColorPalette, Typography
)


class PerformanceMetrics(VisualizationComponent):
    """
    Real-time performance metrics panel for network visualization.
    
    Displays 12 key metrics in a 3x4 grid layout with:
    - Live value updates
    - Trend indicators (up/down/stable arrows)
    - 95% confidence intervals from historical data
    - Color-coded health status
    - Moving averages for smoother trends
    """
    
    # Metric configuration: (label, unit, good_threshold, warning_threshold, higher_is_better)
    METRIC_CONFIG = {
        'pdr': ('Packet Delivery Ratio', '%', 85, 70, True),
        'avg_delay': ('Average Delay', 'ms', 50, 100, False),
        'connectivity': ('Connectivity', '%', 90, 70, True),
        'residual_energy': ('Residual Energy', '%', 50, 20, True),
        'packet_loss': ('Packet Loss', '%', 15, 30, False),
        'network_lifetime': ('Network Lifetime', 'frames', None, None, True),
        'packets_delivered': ('Delivered Packets', '#', None, None, True),
        'packets_dropped': ('Dropped Packets', '#', None, None, False),
        'avg_hop_count': ('Average Hop Count', 'hops', None, None, False),
        'routing_efficiency': ('Routing Efficiency', '%', 80, 60, True),
        'security_score': ('Security Score', '%', 80, 60, True),
        'node_survival_rate': ('Node Survival Rate', '%', 80, 60, True),
    }
    
    MOVING_AVERAGE_WINDOW = 10
    CONFIDENCE_LEVEL = 0.95
    
    def __init__(self, ax: plt.Axes, title: str = "Performance Metrics"):
        """
        Initialize performance metrics panel.
        
        Args:
            ax: Matplotlib axes for the panel
            title: Panel title
        """
        super().__init__(ax, title)
        self.metric_texts = {}  # {metric_key: text_artist}
        self.metric_history = {}  # {metric_key: List[float]}
        self.previous_values = {}  # {metric_key: float} for trend detection
        self.metric_order = list(self.METRIC_CONFIG.keys())
        self.grid_rows = 4
        self.grid_cols = 3
        
    def initialize(self):
        """Set up the metrics grid and configure the axis."""
        self.ax.set_xlim(0, self.grid_cols)
        self.ax.set_ylim(0, self.grid_rows)
        self.ax.invert_yaxis()
        
        # Remove ticks and spines for clean appearance
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Set title
        self.set_title(self.title)
        
        # Add subtle grid background
        for i in range(self.grid_rows + 1):
            self.ax.axhline(y=i, color=ColorPalette.GRID_COLOR, 
                           linewidth=0.5, alpha=0.3, zorder=0)
        for i in range(self.grid_cols + 1):
            self.ax.axvline(x=i, color=ColorPalette.GRID_COLOR,
                           linewidth=0.5, alpha=0.3, zorder=0)
        
        # Initialize metric history tracking
        for metric_key in self.metric_order:
            self.metric_history[metric_key] = []
            self.previous_values[metric_key] = None
        
        # Create text artists for each metric
        for idx, metric_key in enumerate(self.metric_order):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            # Position text in center of grid cell
            x = col + 0.5
            y = row + 0.5
            
            text_artist = self.ax.text(
                x, y, '',
                fontsize=Typography.SIZE_LABEL,
                ha='center', va='center',
                color=ColorPalette.TEXT_PRIMARY,
                fontweight=Typography.WEIGHT_BOLD,
                zorder=10
            )
            self.metric_texts[metric_key] = text_artist
            self.artists.append(text_artist)
    
    def update(self, frame: int, state: Dict[str, Any]) -> List[Any]:
        """
        Update all metrics with current state values.
        
        Args:
            frame: Current frame number
            state: Simulation state dictionary containing metric values
        
        Returns:
            List of matplotlib artists that were updated
        """
        updated_artists = []
        
        for metric_key in self.metric_order:
            # Extract metric value from state
            value = self._extract_metric_value(metric_key, state)
            
            if value is not None:
                # Update history for moving average and CI calculation
                self.metric_history[metric_key].append(value)
                if len(self.metric_history[metric_key]) > self.MOVING_AVERAGE_WINDOW:
                    self.metric_history[metric_key].pop(0)
                
                # Compute display value using moving average
                display_value = np.mean(self.metric_history[metric_key])
                
                # Format and display the metric
                metric_text = self._format_metric_display(
                    metric_key, display_value, value
                )
                
                # Update text artist
                text_artist = self.metric_texts[metric_key]
                text_artist.set_text(metric_text)
                
                # Update color based on value
                color = self._get_metric_color(metric_key, display_value)
                text_artist.set_color(color)
                
                # Update previous value for trend detection
                self.previous_values[metric_key] = display_value
                
                updated_artists.append(text_artist)
        
        return updated_artists
    
    def _extract_metric_value(self, metric_key: str, state: Dict[str, Any]) -> Optional[float]:
        """
        Extract metric value from simulation state.
        
        Args:
            metric_key: Key for the metric (e.g., 'pdr', 'avg_delay')
            state: Simulation state dictionary
        
        Returns:
            Metric value or None if not available
        """
        # Handle special cases and state keys
        if metric_key == 'pdr':
            return state.get('pdr', None)
        elif metric_key == 'avg_delay':
            return state.get('average_delay', None)
        elif metric_key == 'connectivity':
            return state.get('connectivity', None)
        elif metric_key == 'residual_energy':
            energy = state.get('residual_energy', None)
            if isinstance(energy, (list, np.ndarray)):
                return np.mean(energy) * 100 if len(energy) > 0 else None
            return energy
        elif metric_key == 'packet_loss':
            return state.get('packet_loss', None)
        elif metric_key == 'network_lifetime':
            return float(state.get('network_lifetime', 0))
        elif metric_key == 'packets_delivered':
            return float(state.get('packets_delivered', 0))
        elif metric_key == 'packets_dropped':
            return float(state.get('packets_dropped', 0))
        elif metric_key == 'avg_hop_count':
            return state.get('average_hop_count', None)
        elif metric_key == 'routing_efficiency':
            return state.get('routing_efficiency', None)
        elif metric_key == 'security_score':
            return state.get('security_score', None)
        elif metric_key == 'node_survival_rate':
            return state.get('node_survival_rate', None)
        return None
    
    def _get_metric_color(self, metric_key: str, value: float) -> str:
        """
        Determine color for metric based on value and thresholds.
        
        Args:
            metric_key: Key for the metric
            value: Current metric value
        
        Returns:
            Color string (hex or named)
        """
        if metric_key not in self.METRIC_CONFIG:
            return ColorPalette.TEXT_PRIMARY
        
        label, unit, good_threshold, warning_threshold, higher_is_better = \
            self.METRIC_CONFIG[metric_key]
        
        # Metrics without thresholds
        if good_threshold is None or warning_threshold is None:
            return ColorPalette.TEXT_PRIMARY
        
        # Apply thresholds
        if higher_is_better:
            if value >= good_threshold:
                return ColorPalette.HEALTHY_NODE  # Green
            elif value >= warning_threshold:
                return '#FFD700'  # Yellow/Gold
            else:
                return ColorPalette.INFECTED_NODE  # Red
        else:
            # Lower is better (delay, packet loss)
            # good_threshold < warning_threshold for lower-is-better metrics
            if value <= good_threshold:
                return ColorPalette.HEALTHY_NODE  # Green
            elif value <= warning_threshold:
                return '#FFD700'  # Yellow/Gold
            else:
                return ColorPalette.INFECTED_NODE  # Red
    
    def _format_metric_display(self, metric_key: str, display_value: float, 
                               current_value: float) -> str:
        """
        Format metric for display with label, value, trend, and CI.
        
        Format: "[Label] [Value] [Trend] [95% CI if available]"
        Example: "PDR: 87.3% ↑ [84-91%]"
        
        Args:
            metric_key: Key for the metric
            display_value: Moving average value for display
            current_value: Current (raw) value
        
        Returns:
            Formatted metric string
        """
        label, unit, _, _, _ = self.METRIC_CONFIG[metric_key]
        
        # Format value with appropriate precision
        if unit == '%' or metric_key in ['pdr', 'connectivity', 'packet_loss', 
                                          'routing_efficiency', 'security_score', 
                                          'node_survival_rate', 'residual_energy']:
            formatted_value = f"{display_value:.1f}{unit}"
        elif unit == 'ms':
            formatted_value = f"{display_value:.1f}{unit}"
        elif unit == 'hops':
            formatted_value = f"{display_value:.2f}{unit}"
        else:
            # Count values (packets, frames)
            formatted_value = f"{int(display_value)}{unit}"
        
        # Add trend indicator
        trend = self._get_trend_indicator(metric_key, display_value)
        
        # Add confidence interval if available
        ci = self._get_confidence_interval(metric_key)
        ci_str = f" [{ci}]" if ci else ""
        
        # Abbreviate label for compact display
        label_short = self._abbreviate_label(label)
        
        return f"{label_short}: {formatted_value} {trend}{ci_str}"
    
    def _get_trend_indicator(self, metric_key: str, current_value: float) -> str:
        """
        Get trend arrow based on previous value.
        
        Args:
            metric_key: Key for the metric
            current_value: Current value
        
        Returns:
            Trend string: "↑" (improving), "↓" (degrading), "→" (stable)
        """
        if self.previous_values[metric_key] is None:
            return "→"
        
        prev_value = self.previous_values[metric_key]
        _, _, _, _, higher_is_better = self.METRIC_CONFIG[metric_key]
        
        # Determine if metric is improving/degrading
        if higher_is_better:
            if current_value > prev_value * 1.02:
                return "↑"
            elif current_value < prev_value * 0.98:
                return "↓"
        else:
            if current_value < prev_value * 0.98:
                return "↑"
            elif current_value > prev_value * 1.02:
                return "↓"
        
        return "→"
    
    def _get_confidence_interval(self, metric_key: str) -> Optional[str]:
        """
        Compute 95% confidence interval from metric history.
        
        Args:
            metric_key: Key for the metric
        
        Returns:
            CI string in format "min-max" or None if insufficient data
        """
        history = self.metric_history.get(metric_key, [])
        
        if len(history) < 3:
            return None
        
        # Compute 95% CI using sample standard deviation
        mean = np.mean(history)
        std = np.std(history, ddof=1)
        sem = std / np.sqrt(len(history))
        ci_margin = 1.96 * sem  # 95% CI for normal distribution
        
        ci_lower = max(0, mean - ci_margin)
        ci_upper = min(100, mean + ci_margin) if '%' in self.METRIC_CONFIG[metric_key][1] else mean + ci_margin
        
        # Format based on metric type
        if self.METRIC_CONFIG[metric_key][1] == '%':
            return f"{ci_lower:.0f}-{ci_upper:.0f}%"
        elif self.METRIC_CONFIG[metric_key][1] == 'ms':
            return f"{ci_lower:.1f}-{ci_upper:.1f}ms"
        else:
            return f"{ci_lower:.0f}-{ci_upper:.0f}"
    
    def _abbreviate_label(self, label: str) -> str:
        """
        Create abbreviation for metric label for compact display.
        
        Args:
            label: Full metric label
        
        Returns:
            Abbreviated label (typically 3-4 characters)
        """
        abbreviations = {
            'Packet Delivery Ratio': 'PDR',
            'Average Delay': 'Delay',
            'Connectivity': 'Conn',
            'Residual Energy': 'Energy',
            'Packet Loss': 'Loss',
            'Network Lifetime': 'Lifetime',
            'Delivered Packets': 'Delivered',
            'Dropped Packets': 'Dropped',
            'Average Hop Count': 'Hops',
            'Routing Efficiency': 'R-Eff',
            'Security Score': 'Security',
            'Node Survival Rate': 'Survival',
        }
        return abbreviations.get(label, label[:4])
    
    def _compute_moving_average(self, metric_key: str, window: int = None) -> Optional[float]:
        """
        Compute moving average for a metric.
        
        Args:
            metric_key: Key for the metric
            window: Window size for moving average (default: MOVING_AVERAGE_WINDOW)
        
        Returns:
            Moving average value or None if insufficient data
        """
        if window is None:
            window = self.MOVING_AVERAGE_WINDOW
        
        history = self.metric_history.get(metric_key, [])
        if len(history) == 0:
            return None
        
        # Use up to window size
        window_data = history[-window:]
        return np.mean(window_data)
    
    def _add_error_bars(self, metric_key: str, x: float, y: float, 
                        value: float, ax: Optional[plt.Axes] = None) -> Optional[Any]:
        """
        Add error bars (95% CI) to a metric display.
        
        Args:
            metric_key: Key for the metric
            x: X coordinate
            y: Y coordinate
            value: Current value
            ax: Axes object (defaults to self.ax)
        
        Returns:
            Error bar artist or None
        """
        if ax is None:
            ax = self.ax
        
        history = self.metric_history.get(metric_key, [])
        if len(history) < 3:
            return None
        
        std = np.std(history, ddof=1)
        sem = std / np.sqrt(len(history))
        ci_margin = 1.96 * sem
        
        # Could add visual error bars here if needed
        return None
    
    def _add_trend_indicator(self, metric_key: str, x: float, y: float,
                             value: float, ax: Optional[plt.Axes] = None) -> Optional[Any]:
        """
        Add trend indicator (arrow patch) to metric display.
        
        Args:
            metric_key: Key for the metric
            x: X coordinate
            y: Y coordinate
            value: Current value
            ax: Axes object (defaults to self.ax)
        
        Returns:
            Patch object or None
        """
        if ax is None:
            ax = self.ax
        
        trend = self._get_trend_indicator(metric_key, value)
        
        # Trend indicators are shown as part of text, not separate patches
        return None
    
    def clear_history(self):
        """Clear all metric history (useful for reset/restart)."""
        for metric_key in self.metric_order:
            self.metric_history[metric_key] = []
            self.previous_values[metric_key] = None


def demo_performance_metrics():
    """
    Demonstration of the PerformanceMetrics panel.
    
    Creates a simple figure with the metrics panel and simulates updates
    with random data.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.patch.set_facecolor(ColorPalette.BACKGROUND)
    ax.set_facecolor(ColorPalette.BACKGROUND)
    
    metrics = PerformanceMetrics(ax, title="Network Performance Metrics")
    metrics.initialize()
    
    # Simulate 50 frames of data
    np.random.seed(42)
    for frame in range(50):
        # Create mock state with realistic values
        state = {
            'pdr': 85 + np.random.normal(0, 3),
            'average_delay': 25 + np.random.normal(0, 5),
            'connectivity': 92 + np.random.normal(0, 2),
            'residual_energy': 0.60 + np.random.normal(0, 0.05),
            'packet_loss': 8 + np.random.normal(0, 2),
            'network_lifetime': 500 + frame * 5,
            'packets_delivered': 450 + frame * 3,
            'packets_dropped': 50 + frame,
            'average_hop_count': 2.5 + np.random.normal(0, 0.3),
            'routing_efficiency': 85 + np.random.normal(0, 3),
            'security_score': 88 + np.random.normal(0, 2),
            'node_survival_rate': 95 + np.random.normal(0, 1),
        }
        
        metrics.update(frame, state)
    
    plt.tight_layout()
    plt.savefig('performance_metrics_demo.png', dpi=100, facecolor=ColorPalette.BACKGROUND)
    print("Demo figure saved to 'performance_metrics_demo.png'")
    plt.close()


if __name__ == '__main__':
    Typography.configure_mpl()
    demo_performance_metrics()
