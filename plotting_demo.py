#!/usr/bin/env python3
"""
Demonstration of the new plotting capabilities for sensitivity analysis

This script shows how to use the flexible plotting templates for
creating custom 2D and 3D visualizations.
"""

import sys
import os
import numpy as np

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plotting_templates import PlottingTemplates
from sensitivity_analysis import SensitivityAnalyzer

def create_sample_results():
    """Create sample sensitivity analysis results for demonstration"""
    
    # Simulate some sensitivity analysis results
    results = []
    parameters = ['Qk', 'H_b', 'H_t_40', 'H_t_20', 'Handling_time']
    
    for param in parameters:
        for variation in [0, 10, 20, 30, 50]:
            # Simulate results with some realistic relationships
            base_cost = 15000
            if param == 'Qk':
                cost_multiplier = 1 - variation * 0.01  # Higher capacity -> lower cost
            elif param in ['H_b', 'H_t_40', 'H_t_20']:
                cost_multiplier = 1 + variation * 0.02  # Higher costs -> higher total cost
            else:
                cost_multiplier = 1 + variation * 0.005  # Small impact
            
            final_cost = base_cost * cost_multiplier + np.random.normal(0, 500)
            
            # Related metrics
            improvement_pct = np.random.uniform(5, 25)
            barge_utilization = np.random.uniform(40, 90)
            containers_on_barges = int(np.random.uniform(80, 150))
            barges_used = int(np.random.uniform(3, 8))
            
            result = {
                'parameter': param,
                'variation_pct': variation,
                'final_cost': final_cost,
                'improvement_pct': improvement_pct,
                'barge_utilization_pct': barge_utilization,
                'containers_on_barges': containers_on_barges,
                'barges_used': barges_used,
                'C_actual': 160,
                'success': True
            }
            results.append(result)
    
    return results

def demonstrate_plotting_templates():
    """Demonstrate the PlottingTemplates class directly"""
    
    print("🎨 Demonstrating PlottingTemplates class...")
    
    plotter = PlottingTemplates()
    
    # 1. Simple 2D plot
    print("Creating simple 2D plot...")
    x = [1, 2, 3, 4, 5]
    y = [10, 15, 13, 18, 16]
    
    fig1, ax1 = plotter.plot_2d(
        x, y, 
        x_label="Parameter Variation (%)",
        y_label="Cost (€)",
        title="Cost vs Parameter Variation",
        plot_type='both'
    )
    
    # 2. Multiple series 2D plot
    print("Creating multiple series 2D plot...")
    x_dict = {
        'Method A': [0, 10, 20, 30, 50],
        'Method B': [0, 10, 20, 30, 50],
        'Method C': [0, 10, 20, 30, 50]
    }
    y_dict = {
        'Method A': [15000, 15500, 16000, 16800, 18000],
        'Method B': [15200, 15300, 15600, 16200, 17500], 
        'Method C': [14800, 15100, 15800, 16500, 17800]
    }
    
    fig2, ax2 = plotter.plot_2d(
        x_dict, y_dict,
        x_label="Parameter Variation (%)",
        y_label="Final Cost (€)",
        title="Method Comparison",
        plot_type='both'  
    )
    
    # 3. 3D scatter plot
    print("Creating 3D scatter plot...")
    np.random.seed(42)
    x = np.random.uniform(14000, 18000, 50)  # Final cost
    y = np.random.uniform(5, 25, 50)         # Improvement %
    z = np.random.uniform(40, 90, 50)        # Utilization %
    colors = np.random.randint(0, 5, 50)     # Parameter types
    
    fig3, ax3 = plotter.plot_3d(
        x, y, z, color_data=colors,
        x_label="Final Cost (€)",
        y_label="Improvement (%)",
        z_label="Utilization (%)",
        title="3D Parameter Analysis",
        colorbar_label="Parameter Type"
    )
    
    print("✅ PlottingTemplates demonstration complete!")
    print("Close the plots to continue...")
    plotter.show_all_plots()

def demonstrate_sensitivity_analyzer():
    """Demonstrate the enhanced SensitivityAnalyzer with plotting templates"""
    
    print("🔍 Demonstrating SensitivityAnalyzer with plotting templates...")
    
    # Create analyzer and load sample results
    analyzer = SensitivityAnalyzer()
    analyzer.results = create_sample_results()
    
    print(f"📊 Loaded {len(analyzer.results)} sample results")
    
    # Show available variables for plotting
    variables = analyzer.get_available_variables()
    print(f"📈 Available variables for plotting: {variables}")
    
    # 1. Create a 2D plot using the template
    print("Creating 2D plot: Final Cost vs Improvement...")
    fig1, ax1 = analyzer.plot_2d_custom(
        'variation_pct', 'final_cost',
        plot_type='both',
        title="Final Cost vs Parameter Variation"
    )
    
    # 2. Create a 3D plot
    print("Creating 3D plot: Cost vs Improvement vs Utilization...")
    fig2, ax2 = analyzer.plot_3d_custom(
        'final_cost', 'improvement_pct', 'barge_utilization_pct',
        color_var='parameter',
        title="3D Analysis: Cost vs Improvement vs Utilization"
    )
    
    # 3. Use the parameter comparison method
    print("Creating parameter comparison plot...")
    fig3, ax3 = analyzer.plot_variable_comparison(
        'variation_pct', 'final_cost', plot_type='2d',
        title_prefix="Parameter Sensitivity"
    )
    
    # 4. Create a 3D parameter comparison
    print("Creating 3D parameter comparison...")
    fig4, ax4 = analyzer.plot_variable_comparison(
        'final_cost', 'improvement_pct', 'barge_utilization_pct', 
        plot_type='3d',
        title_prefix="3D Parameter Analysis"
    )
    
    print("✅ SensitivityAnalyzer demonstration complete!")
    print("Close the plots to finish...")
    analyzer.show_plots()

def main():
    """Main demonstration function"""
    
    print("🚀 Plotting Templates Demonstration")
    print("=" * 50)
    
    # Ask user which demonstration to run
    print("Choose demonstration:")
    print("1. PlottingTemplates class only")
    print("2. SensitivityAnalyzer with plotting templates")
    print("3. Both demonstrations")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice in ['1', '3']:
        demonstrate_plotting_templates()
        input("Press Enter to continue...")
    
    if choice in ['2', '3']:  
        demonstrate_sensitivity_analyzer()
    
    print("\n🎉 Demonstration complete!")
    print("\nExample usage in your code:")
    print("=" * 30)
    print("# Create analyzer")
    print("analyzer = SensitivityAnalyzer()")
    print("# Run analysis...")
    print("results = analyzer.run_quick_analysis()")
    print("")
    print("# Plot any variables")
    print("analyzer.plot_2d_custom('final_cost', 'improvement_pct')")
    print("analyzer.plot_3d_custom('final_cost', 'improvement_pct', 'barge_utilization_pct')")
    print("")
    print("# Or use the flexible comparison")
    print("analyzer.plot_variable_comparison('var1', 'var2', 'var3', plot_type='3d')")
    print("analyzer.show_plots()")

if __name__ == "__main__":
    main()