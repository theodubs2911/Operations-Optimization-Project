#!/usr/bin/env python3
"""
Plotting Utilities for Container Allocation Optimization

This module provides flexible 2D and 3D plotting templates for sensitivity analysis
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class PlottingTemplates:
    """Flexible plotting templates for 2D and 3D visualizations"""
    
    def __init__(self):
        """Initialize plotting templates with default styling"""
        self.default_figsize_2d = (10, 8)
        self.default_figsize_3d = (12, 9)
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    def plot_2d(self, x_data, y_data, labels=None, x_label="X Variable", y_label="Y Variable", 
                title="2D Analysis", plot_type='line', figsize=None, colors=None, 
                markers=None, show_grid=True, legend_loc='best', save_path=None):
        """
        Flexible 2D plotting template
        
        Parameters:
        -----------
        x_data : list or dict
            X-axis data. If dict, keys are series names
        y_data : list or dict  
            Y-axis data. If dict, keys are series names
        labels : list, optional
            Series labels for legend
        x_label : str
            X-axis label
        y_label : str
            Y-axis label
        title : str
            Plot title
        plot_type : str
            'line', 'scatter', 'bar', 'both'
        figsize : tuple, optional
            Figure size (width, height)
        colors : list, optional
            Custom colors for series
        markers : list, optional
            Custom markers for series
        show_grid : bool
            Whether to show grid
        legend_loc : str
            Legend location
        save_path : str, optional
            Path to save the plot
        
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        
        if figsize is None:
            figsize = self.default_figsize_2d
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Handle different data formats
        if isinstance(x_data, dict) and isinstance(y_data, dict):
            # Multiple series from dictionaries
            series_names = list(x_data.keys())
            if labels is None:
                labels = series_names
                
            for i, series in enumerate(series_names):
                x_vals = x_data[series]
                y_vals = y_data[series]
                color = colors[i] if colors and i < len(colors) else self.color_palette[i % len(self.color_palette)]
                marker = markers[i] if markers and i < len(markers) else 'o'
                
                if plot_type == 'line':
                    ax.plot(x_vals, y_vals, color=color, label=labels[i], alpha=0.7, linewidth=2)
                elif plot_type == 'scatter':
                    ax.scatter(x_vals, y_vals, color=color, label=labels[i], alpha=0.7, s=50)
                elif plot_type == 'both':
                    ax.plot(x_vals, y_vals, color=color, label=labels[i], alpha=0.7, linewidth=2, marker=marker)
                    
        elif isinstance(x_data, list) and isinstance(y_data, list):
            # Single series
            color = colors[0] if colors else self.color_palette[0]
            marker = markers[0] if markers else 'o'
            label = labels[0] if labels else 'Data'
            
            if plot_type == 'line':
                ax.plot(x_data, y_data, color=color, label=label, alpha=0.7, linewidth=2)
            elif plot_type == 'scatter':
                ax.scatter(x_data, y_data, color=color, label=label, alpha=0.7, s=50)
            elif plot_type == 'both':
                ax.plot(x_data, y_data, color=color, label=label, alpha=0.7, linewidth=2, marker=marker)
            elif plot_type == 'bar':
                ax.bar(range(len(x_data)), y_data, color=color, alpha=0.7)
                ax.set_xticks(range(len(x_data)))
                ax.set_xticklabels(x_data)
        
        # Styling
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if show_grid:
            ax.grid(True, alpha=0.3)
            
        if labels and plot_type != 'bar':
            ax.legend(loc=legend_loc)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, ax
    
    def plot_3d(self, x_data, y_data, z_data, color_data=None, 
                x_label="X Variable", y_label="Y Variable", z_label="Z Variable",
                title="3D Analysis", plot_type='scatter', figsize=None, cmap='viridis',
                alpha=0.7, size=60, save_path=None, colorbar_label=None):
        """
        Flexible 3D plotting template
        
        Parameters:
        -----------
        x_data : list or array
            X-axis data
        y_data : list or array
            Y-axis data  
        z_data : list or array
            Z-axis data
        color_data : list or array, optional
            Data for color mapping
        x_label : str
            X-axis label
        y_label : str
            Y-axis label
        z_label : str
            Z-axis label
        title : str
            Plot title
        plot_type : str
            'scatter', 'surface', 'wireframe'
        figsize : tuple, optional
            Figure size (width, height)
        cmap : str
            Colormap name
        alpha : float
            Transparency level
        size : int
            Point size for scatter plots
        save_path : str, optional
            Path to save the plot
        colorbar_label : str, optional
            Label for colorbar
        
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        
        if figsize is None:
            figsize = self.default_figsize_3d
            
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if plot_type == 'scatter':
            if color_data is not None:
                scatter = ax.scatter(x_data, y_data, z_data, c=color_data, 
                                   cmap=cmap, alpha=alpha, s=size)
                if colorbar_label:
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
                    cbar.set_label(colorbar_label)
                else:
                    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            else:
                ax.scatter(x_data, y_data, z_data, alpha=alpha, s=size)
                
        elif plot_type == 'surface':
            # For surface plots, data should be meshgrid format
            if isinstance(x_data, (list, np.ndarray)) and len(np.array(x_data).shape) == 1:
                # Convert 1D arrays to meshgrid
                X, Y = np.meshgrid(x_data, y_data)
                Z = np.array(z_data).reshape(len(y_data), len(x_data))
            else:
                X, Y, Z = x_data, y_data, z_data
                
            surface = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha)
            if colorbar_label:
                cbar = plt.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
                cbar.set_label(colorbar_label)
            else:
                plt.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
                
        elif plot_type == 'wireframe':
            if isinstance(x_data, (list, np.ndarray)) and len(np.array(x_data).shape) == 1:
                X, Y = np.meshgrid(x_data, y_data)
                Z = np.array(z_data).reshape(len(y_data), len(x_data))
            else:
                X, Y, Z = x_data, y_data, z_data
                
            ax.plot_wireframe(X, Y, Z, alpha=alpha)
        
        # Styling
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_zlabel(z_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, ax
    
    def plot_parameter_comparison(self, results_data, var1, var2, var3=None, 
                                plot_type='2d', title_prefix="Parameter Analysis"):
        """
        Plot comparison of variables from results data
        
        Parameters:
        -----------
        results_data : list of dicts
            Results from sensitivity analysis
        var1 : str
            Variable name for X-axis (or color in 3D)
        var2 : str  
            Variable name for Y-axis
        var3 : str, optional
            Variable name for Z-axis (3D only)
        plot_type : str
            '2d' or '3d'
        title_prefix : str
            Prefix for plot title
            
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        
        if not results_data:
            print("No data to plot")
            return None, None
            
        # Extract data
        x_data = [r.get(var1, 0) for r in results_data]
        y_data = [r.get(var2, 0) for r in results_data]
        
        if plot_type == '3d' and var3:
            z_data = [r.get(var3, 0) for r in results_data]
            
            # Use parameter names as color coding if available
            color_data = None
            if 'parameter' in results_data[0]:
                unique_params = list(set(r['parameter'] for r in results_data))
                param_to_num = {param: i for i, param in enumerate(unique_params)}
                color_data = [param_to_num[r['parameter']] for r in results_data]
            
            title = f"{title_prefix}: {var1} vs {var2} vs {var3}"
            
            return self.plot_3d(
                x_data, y_data, z_data, color_data=color_data,
                x_label=var1.replace('_', ' ').title(),
                y_label=var2.replace('_', ' ').title(), 
                z_label=var3.replace('_', ' ').title(),
                title=title,
                colorbar_label='Parameter Type'
            )
            
        else:
            # 2D plot
            # Group by parameter if available
            if 'parameter' in results_data[0]:
                param_data = {}
                for result in results_data:
                    param = result.get('parameter', 'unknown')
                    if param not in param_data:
                        param_data[param] = {'x': [], 'y': []}
                    param_data[param]['x'].append(result.get(var1, 0))
                    param_data[param]['y'].append(result.get(var2, 0))
                
                x_dict = {param: data['x'] for param, data in param_data.items()}
                y_dict = {param: data['y'] for param, data in param_data.items()}
                
                title = f"{title_prefix}: {var2} vs {var1}"
                
                return self.plot_2d(
                    x_dict, y_dict,
                    x_label=var1.replace('_', ' ').title(),
                    y_label=var2.replace('_', ' ').title(),
                    title=title,
                    plot_type='both'
                )
            else:
                title = f"{title_prefix}: {var2} vs {var1}"
                
                return self.plot_2d(
                    x_data, y_data,
                    x_label=var1.replace('_', ' ').title(),
                    y_label=var2.replace('_', ' ').title(),
                    title=title,
                    plot_type='scatter'
                )
    
    def show_all_plots(self):
        """Display all created plots"""
        plt.show()
    
    def close_all_plots(self):
        """Close all plots"""
        plt.close('all')

# Example usage functions
def example_2d_usage():
    """Example of how to use 2D plotting template"""
    plotter = PlottingTemplates()
    
    # Single series example
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 1, 5, 3]
    
    fig, ax = plotter.plot_2d(x, y, x_label="Iteration", y_label="Cost", 
                             title="Cost vs Iteration", plot_type='both')
    
    # Multiple series example
    x_dict = {
        'Method A': [1, 2, 3, 4, 5],
        'Method B': [1, 2, 3, 4, 5]
    }
    y_dict = {
        'Method A': [2, 4, 1, 5, 3],
        'Method B': [1, 3, 2, 4, 4]
    }
    
    fig2, ax2 = plotter.plot_2d(x_dict, y_dict, x_label="Parameter Value", 
                               y_label="Performance", title="Method Comparison",
                               plot_type='both')
    
    plotter.show_all_plots()

def example_3d_usage():
    """Example of how to use 3D plotting template"""
    plotter = PlottingTemplates()
    
    # Generate sample data
    x = np.random.randn(50)
    y = np.random.randn(50) 
    z = x**2 + y**2 + np.random.randn(50)*0.1
    colors = np.random.randint(0, 3, 50)
    
    fig, ax = plotter.plot_3d(x, y, z, color_data=colors,
                             x_label="Parameter 1", y_label="Parameter 2", 
                             z_label="Objective Value", title="3D Parameter Space",
                             colorbar_label="Method")
    
    plotter.show_all_plots()

if __name__ == "__main__":
    # Run examples
    print("Running 2D example...")
    example_2d_usage()
    
    print("Running 3D example...")  
    example_3d_usage()