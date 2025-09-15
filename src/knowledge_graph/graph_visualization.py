#!/usr/bin/env python3
"""
HSN Knowledge Graph Visualization - Phase 2.3
Advanced visualization capabilities for the HSN knowledge graph
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# File paths
OUTPUT_DIR = Path("output")
DATA_DIR = OUTPUT_DIR / "data"
MODELS_DIR = OUTPUT_DIR / "models"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
GRAPH_FILE = MODELS_DIR / "hsn_knowledge_graph.pkl"

VISUALIZATIONS_DIR.mkdir(exist_ok=True)

@dataclass
class VisualizationConfig:
    """Configuration for graph visualizations"""
    width: int = 1200
    height: int = 800
    node_size_range: Tuple[int, int] = (20, 50)
    edge_width_range: Tuple[float, float] = (1, 3)
    color_scheme: str = "default"
    layout_algorithm: str = "spring"

class HSNGraphVisualizer:
    """
    Advanced visualization capabilities for HSN Knowledge Graph
    """

    def __init__(self, graph_file: Path = GRAPH_FILE):
        self.graph = None
        self.config = VisualizationConfig()
        self.load_graph(graph_file)

    def load_graph(self, graph_file: Path) -> bool:
        """Load the NetworkX graph from pickle file"""
        try:
            with open(graph_file, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"SUCCESS: Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load graph: {str(e)}")
            return False

    def create_interactive_visualization(self,
                                       output_file: Optional[Path] = None,
                                       show_hierarchy: bool = True) -> go.Figure:
        """
        Create an interactive Plotly visualization of the knowledge graph

        Args:
            output_file: Optional path to save HTML file
            show_hierarchy: Whether to show hierarchical layout

        Returns:
            Plotly figure object
        """
        if not self.graph:
            raise ValueError("Graph not loaded")

        # Choose layout
        if show_hierarchy:
            pos = self._create_hierarchical_layout()
        else:
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)

        # Create node traces
        node_traces = self._create_node_traces(pos)

        # Create edge traces
        edge_traces = self._create_edge_traces(pos)

        # Combine all traces
        all_traces = edge_traces + node_traces

        # Create figure
        fig = go.Figure(data=all_traces)

        # Update layout
        fig.update_layout(
            title="HSN Knowledge Graph - Interactive Visualization",
            title_x=0.5,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            width=self.config.width,
            height=self.config.height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        # Add interactive features
        fig.update_traces(
            hovertemplate="<b>%{text}</b><br>" +
                         "Type: %{customdata[0]}<br>" +
                         "Code: %{customdata[1]}<br>" +
                         "Connections: %{customdata[2]}<extra></extra>"
        )

        if output_file:
            fig.write_html(str(output_file))
            print(f"SUCCESS: Interactive visualization saved to {output_file}")

        return fig

    def _create_hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create a hierarchical layout for the graph"""
        pos = {}

        # Group nodes by hierarchy level
        level_nodes = {
            'chapter': [],
            'heading': [],
            'subheading': [],
            'hsn': []
        }

        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            if node_type in level_nodes:
                level_nodes[node_type].append(node_id)

        # Assign positions by level
        level_heights = {'chapter': 3, 'heading': 2, 'subheading': 1, 'hsn': 0}
        level_widths = {}

        for level, nodes in level_nodes.items():
            if nodes:
                level_widths[level] = len(nodes)
                height = level_heights[level]
                width_step = 2.0 / max(len(nodes), 1)

                for i, node_id in enumerate(nodes):
                    x = -1.0 + i * width_step
                    pos[node_id] = (x, height)

        return pos

    def _create_node_traces(self, pos: Dict[str, Tuple[float, float]]) -> List[go.Scatter]:
        """Create Plotly traces for nodes"""
        traces = []

        # Group nodes by type
        node_groups = {}
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            if node_type not in node_groups:
                node_groups[node_type] = []
            node_groups[node_type].append((node_id, node_data))

        # Color scheme
        colors = {
            'chapter': '#FF6B6B',      # Red
            'heading': '#4ECDC4',      # Teal
            'subheading': '#45B7D1',   # Blue
            'hsn': '#96CEB4',          # Green
            'unknown': '#D4D4D4'       # Gray
        }

        for node_type, nodes in node_groups.items():
            x_coords = []
            y_coords = []
            texts = []
            custom_data = []
            sizes = []

            for node_id, node_data in nodes:
                if node_id in pos:
                    x, y = pos[node_id]
                    x_coords.append(x)
                    y_coords.append(y)

                    # Create hover text
                    code = node_data.get('code', 'N/A')
                    description = node_data.get('description', 'N/A')[:50]
                    title = node_data.get('title', 'N/A')[:30]

                    texts.append(f"{node_type.upper()}: {code}<br>{title}")

                    # Custom data for hover template
                    connections = self.graph.degree(node_id)
                    custom_data.append([node_type, code, connections])

                    # Size based on connections
                    size = min(self.config.node_size_range[1],
                             self.config.node_size_range[0] + connections * 2)
                    sizes.append(size)

            if x_coords:
                trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers+text',
                    name=f"{node_type.title()}s ({len(nodes)})",
                    marker=dict(
                        size=sizes,
                        color=colors.get(node_type, '#D4D4D4'),
                        line=dict(width=2, color='white'),
                        sizemode='diameter'
                    ),
                    text=[node_data.get('code', '') for _, node_data in nodes],
                    textposition="middle center",
                    textfont=dict(size=8, color='white'),
                    hovertext=texts,
                    customdata=custom_data,
                    showlegend=True
                )
                traces.append(trace)

        return traces

    def _create_edge_traces(self, pos: Dict[str, Tuple[float, float]]) -> List[go.Scatter]:
        """Create Plotly traces for edges"""
        traces = []

        # Group edges by type
        edge_groups = {}
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            if key not in edge_groups:
                edge_groups[key] = []
            edge_groups[key].append((source, target, data))

        # Color scheme for edges
        edge_colors = {
            'IS_PARENT_OF': '#FF6B6B',        # Red
            'IS_CHILD_OF': '#4ECDC4',         # Teal
            'BELONGS_TO_CHAPTER': '#45B7D1',  # Blue
            'HAS_SIMILAR_DESCRIPTION': '#96CEB4', # Green
            'SHARES_EXPORT_POLICY': '#FECA57'     # Yellow
        }

        for edge_type, edges in edge_groups.items():
            x_coords = []
            y_coords = []

            for source, target, data in edges:
                if source in pos and target in pos:
                    x0, y0 = pos[source]
                    x1, y1 = pos[target]

                    x_coords.extend([x0, x1, None])
                    y_coords.extend([y0, y1, None])

            if x_coords:
                trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    name=f"{edge_type.replace('_', ' ').title()} ({len(edges)})",
                    line=dict(
                        width=2,
                        color=edge_colors.get(edge_type, '#D4D4D4')
                    ),
                    hoverinfo='skip',
                    showlegend=True
                )
                traces.append(trace)

        return traces

    def create_hierarchical_tree_visualization(self,
                                             root_node: str = None,
                                             output_file: Optional[Path] = None) -> go.Figure:
        """
        Create a hierarchical tree visualization

        Args:
            root_node: Root node to start from (defaults to first chapter)
            output_file: Optional path to save HTML file

        Returns:
            Plotly figure object
        """
        if not self.graph:
            raise ValueError("Graph not loaded")

        # Find root node (first chapter if not specified)
        if not root_node:
            chapters = [n for n in self.graph.nodes() if n.startswith('chapter_')]
            root_node = chapters[0] if chapters else None

        if not root_node or root_node not in self.graph:
            raise ValueError("Valid root node not found")

        # Build tree structure
        tree_data = self._build_tree_structure(root_node)

        # Create sunburst chart
        fig = go.Figure(go.Sunburst(
            labels=tree_data['labels'],
            parents=tree_data['parents'],
            values=tree_data['values'],
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Children: %{value}<extra></extra>',
            maxdepth=4
        ))

        fig.update_layout(
            title=f"HSN Hierarchical Tree - Root: {root_node}",
            title_x=0.5,
            width=self.config.width,
            height=self.config.height
        )

        if output_file:
            fig.write_html(str(output_file))
            print(f"SUCCESS: Hierarchical tree visualization saved to {output_file}")

        return fig

    def _build_tree_structure(self, root_node: str) -> Dict[str, List]:
        """Build tree structure for sunburst visualization"""
        labels = []
        parents = []
        values = []

        def add_node(node_id: str, parent: str = ""):
            node_data = self.graph.nodes[node_id]
            code = node_data.get('code', 'N/A')
            desc = node_data.get('description', '')[:30]

            labels.append(f"{code}<br>{desc}")
            parents.append(parent)
            values.append(1)  # Each node has value 1

            # Add children
            children = []
            for successor in self.graph.successors(node_id):
                edge_data = self.graph.get_edge_data(node_id, successor)
                for rel_type in edge_data.keys():
                    if rel_type == 'IS_PARENT_OF':
                        children.append(successor)

            for child in children:
                add_node(child, code)

        add_node(root_node)
        return {'labels': labels, 'parents': parents, 'values': values}

    def create_statistics_dashboard(self, output_file: Optional[Path] = None) -> go.Figure:
        """
        Create a comprehensive statistics dashboard

        Args:
            output_file: Optional path to save HTML file

        Returns:
            Plotly figure with dashboard
        """
        if not self.graph:
            raise ValueError("Graph not loaded")

        # Get statistics
        stats = self._calculate_detailed_statistics()

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Node Distribution', 'Relationship Distribution',
                          'Connectivity Analysis', 'Hierarchy Analysis'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # Node distribution pie chart
        node_types = list(stats['nodes']['by_type'].keys())
        node_counts = list(stats['nodes']['by_type'].values())

        fig.add_trace(
            go.Pie(labels=node_types, values=node_counts, name="Node Types"),
            row=1, col=1
        )

        # Relationship distribution bar chart
        rel_types = list(stats['relationships']['by_type'].keys())
        rel_counts = list(stats['relationships']['by_type'].values())

        fig.add_trace(
            go.Bar(x=rel_types, y=rel_counts, name="Relationship Types",
                  marker_color='lightblue'),
            row=1, col=2
        )

        # Connectivity analysis scatter plot
        degrees = dict(self.graph.degree())
        node_ids = list(degrees.keys())
        degree_values = list(degrees.values())

        fig.add_trace(
            go.Scatter(x=node_ids, y=degree_values, mode='markers',
                      name="Node Degrees", marker=dict(size=8, opacity=0.6)),
            row=2, col=1
        )

        # Hierarchy analysis bar chart
        hierarchy_data = stats['hierarchy']
        hierarchy_labels = ['Max Depth', 'Leaf Nodes', 'Root Nodes']
        hierarchy_values = [
            hierarchy_data['max_depth'],
            hierarchy_data['leaf_nodes'],
            len([n for n in self.graph.nodes() if n.startswith('chapter_')])
        ]

        fig.add_trace(
            go.Bar(x=hierarchy_labels, y=hierarchy_values,
                  name="Hierarchy Metrics", marker_color='lightgreen'),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title="HSN Knowledge Graph Statistics Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )

        # Update axes
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=2)

        if output_file:
            fig.write_html(str(output_file))
            print(f"SUCCESS: Statistics dashboard saved to {output_file}")

        return fig

    def _calculate_detailed_statistics(self) -> Dict[str, Any]:
        """Calculate detailed statistics for the dashboard"""
        stats = {
            'nodes': {
                'total': self.graph.number_of_nodes(),
                'by_type': {}
            },
            'relationships': {
                'total': self.graph.number_of_edges(),
                'by_type': {}
            },
            'connectivity': {
                'average_degree': 0,
                'max_degree': 0,
                'min_degree': float('inf')
            },
            'hierarchy': {
                'max_depth': 0,
                'leaf_nodes': 0,
                'root_nodes': 0
            }
        }

        # Node statistics
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            stats['nodes']['by_type'][node_type] = stats['nodes']['by_type'].get(node_type, 0) + 1

        # Relationship statistics
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            stats['relationships']['by_type'][key] = stats['relationships']['by_type'].get(key, 0) + 1

        # Connectivity statistics
        degrees = dict(self.graph.degree())
        if degrees:
            stats['connectivity']['average_degree'] = sum(degrees.values()) / len(degrees)
            stats['connectivity']['max_degree'] = max(degrees.values())
            stats['connectivity']['min_degree'] = min(degrees.values())

        # Hierarchy statistics
        stats['hierarchy']['max_depth'] = self._calculate_max_depth()
        stats['hierarchy']['leaf_nodes'] = len([n for n in self.graph.nodes()
                                               if self.graph.out_degree(n) == 0])
        stats['hierarchy']['root_nodes'] = len([n for n in self.graph.nodes()
                                               if n.startswith('chapter_')])

        return stats

    def _calculate_max_depth(self) -> int:
        """Calculate the maximum depth of the hierarchy"""
        max_depth = 0
        root_nodes = [n for n in self.graph.nodes() if n.startswith('chapter_')]

        for root in root_nodes:
            try:
                depths = nx.shortest_path_length(self.graph, source=root)
                if depths:
                    max_depth = max(max_depth, max(depths.values()))
            except:
                continue

        return max_depth

    def export_visualizations(self, output_dir: Path = VISUALIZATIONS_DIR) -> Dict[str, str]:
        """
        Export all visualizations in multiple formats

        Args:
            output_dir: Directory to save visualizations

        Returns:
            Dictionary of exported file paths
        """
        exported_files = {}

        try:
            # Interactive visualization
            interactive_file = output_dir / "hsn_graph_interactive.html"
            self.create_interactive_visualization(interactive_file, show_hierarchy=True)
            exported_files["interactive_html"] = str(interactive_file)

            # Hierarchical tree
            tree_file = output_dir / "hsn_hierarchy_tree.html"
            self.create_hierarchical_tree_visualization(output_file=tree_file)
            exported_files["tree_html"] = str(tree_file)

            # Statistics dashboard
            dashboard_file = output_dir / "hsn_statistics_dashboard.html"
            self.create_statistics_dashboard(dashboard_file)
            exported_files["dashboard_html"] = str(dashboard_file)

            # Static visualizations
            static_files = self._create_static_visualizations(output_dir)
            exported_files.update(static_files)

            print(f"SUCCESS: Exported {len(exported_files)} visualization files")
            return exported_files

        except Exception as e:
            print(f"ERROR: Failed to export visualizations: {str(e)}")
            return {}

    def _create_static_visualizations(self, output_dir: Path) -> Dict[str, str]:
        """Create static matplotlib visualizations"""
        exported_files = {}

        try:
            # Create multiple static views
            views = [
                ("spring", "Spring Layout"),
                ("circular", "Circular Layout"),
                ("random", "Random Layout")
            ]

            for layout_name, title in views:
                fig, ax = plt.subplots(figsize=(16, 12))

                if layout_name == "spring":
                    pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
                elif layout_name == "circular":
                    pos = nx.circular_layout(self.graph)
                else:
                    pos = nx.random_layout(self.graph, seed=42)

                # Draw nodes by type
                node_colors = {'chapter': 'red', 'heading': 'orange',
                             'subheading': 'yellow', 'hsn': 'green'}

                for node_type, color in node_colors.items():
                    nodes = [n for n, d in self.graph.nodes(data=True)
                           if d.get('node_type') == node_type]
                    if nodes:
                        nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes,
                                             node_color=color, node_size=300,
                                             alpha=0.7, label=f"{node_type.title()}s")

                # Draw edges
                nx.draw_networkx_edges(self.graph, pos, alpha=0.3, width=1)

                # Add labels for important nodes
                labels = {}
                for node_id, node_data in self.graph.nodes(data=True):
                    if node_data.get('node_type') in ['chapter', 'heading']:
                        labels[node_id] = node_data.get('code', '')

                nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, font_weight='bold')

                plt.title(f"HSN Knowledge Graph - {title}")
                plt.legend()
                plt.axis('off')

                # Save figure
                filename = f"hsn_graph_{layout_name}.png"
                filepath = output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                exported_files[f"{layout_name}_png"] = str(filepath)

            return exported_files

        except Exception as e:
            print(f"ERROR: Failed to create static visualizations: {str(e)}")
            return {}

def main():
    """Main execution function for graph visualization."""
    print("Starting HSN Knowledge Graph Visualization (Phase 2.3)")
    print("=" * 70)

    try:
        # Initialize visualizer
        visualizer = HSNGraphVisualizer()

        # Create interactive visualization
        print("\n1. Creating interactive graph visualization...")
        interactive_file = VISUALIZATIONS_DIR / "hsn_graph_interactive.html"
        interactive_fig = visualizer.create_interactive_visualization(interactive_file, show_hierarchy=True)

        # Create hierarchical tree visualization
        print("\n2. Creating hierarchical tree visualization...")
        tree_file = VISUALIZATIONS_DIR / "hsn_hierarchy_tree.html"
        tree_fig = visualizer.create_hierarchical_tree_visualization(output_file=tree_file)

        # Create statistics dashboard
        print("\n3. Creating statistics dashboard...")
        dashboard_file = VISUALIZATIONS_DIR / "hsn_statistics_dashboard.html"
        dashboard_fig = visualizer.create_statistics_dashboard(dashboard_file)

        # Export all visualizations
        print("\n4. Exporting all visualizations...")
        exported_files = visualizer.export_visualizations(VISUALIZATIONS_DIR)

        # Summary
        print("\n" + "=" * 70)
        print("PHASE 2.3 GRAPH VISUALIZATION COMPLETE")
        print("=" * 70)
        print(f"SUCCESS: Created interactive visualizations")
        print(f"SUCCESS: Generated hierarchical tree views")
        print(f"SUCCESS: Built statistics dashboard")
        print(f"SUCCESS: Exported {len(exported_files)} visualization files")
        print(f"Files saved to: {VISUALIZATIONS_DIR}")
        print("Ready to proceed to Phase 3: RAG System Implementation")
        print("=" * 70)

        # Display sample statistics
        stats = visualizer._calculate_detailed_statistics()
        print("\nGraph Overview:")
        print(f"Nodes: {stats['nodes']['total']} ({stats['nodes']['by_type']})")
        print(f"Relationships: {stats['relationships']['total']} ({stats['relationships']['by_type']})")
        print(f"Hierarchy: Max depth {stats['hierarchy']['max_depth']}, {stats['hierarchy']['leaf_nodes']} leaf nodes")

    except Exception as e:
        print(f"ERROR: Error in graph visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main()