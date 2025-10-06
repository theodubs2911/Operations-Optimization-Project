#!/usr/bin/env python3
"""
Sensitivity Analysis for Container Allocation Optimization

This module provides tools to analyze how parameter changes affect
the meta-heuristic optimization performance.
"""

import sys
import os
import json
from datetime import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Meta_Heuristics import MetaHeuristic
import Greedy_Algo 
from plotting_templates import PlottingTemplates

# Import optimization modules (imported when needed to avoid dependency issues)

class SensitivityAnalyzer:
    """Analyze sensitivity of optimization to parameter changes"""
    
    def __init__(self):
        """Initialize the sensitivity analyzer with base parameters"""
        self.base_params = {
            'Qk': 2500,      # Barge capacity (TEU)
            'H_b': 60,       # Barge hourly cost
            'H_t_40': 65,    # Truck cost 40ft
            'H_t_20': 45,    # Truck cost 20ft
            'Handling_time': 0.4,  # Container handling time
            'travel_times': 1.0,   # Travel time multiplier
            'num_barges': 8, # Number of available barges
            'N': 16,         # Number of terminals  
            'C': 160         # Number of containers
        }
        
        # Parameter variation percentages for continuous parameters
        self.variation_percentages = [0, 10, 20, 30, 50]
        
        # Discrete parameter variations
        self.discrete_variations = {
            'num_barges': [4, 6, 8, 10, 12],
            'N': [8, 12, 16, 20, 24],
            'C': [80, 120, 160, 200, 240]
        }
        
        self.results = []
        
        # Initialize plotting templates
        self.plotter = PlottingTemplates()
        
    def modify_greedy_algo_params(self, param_name, new_value):
        """Modify global parameters in Greedy_Algo module"""
        import Greedy_Algo
        
        if param_name == 'Qk':
            Greedy_Algo.Qk = {k: new_value for k in range(20)}
        elif param_name == 'H_b':
            Greedy_Algo.H_b = {k: new_value for k in range(20)}
        elif param_name == 'H_t_40':
            Greedy_Algo.H_t_40 = new_value
        elif param_name == 'H_t_20':
            Greedy_Algo.H_t_20 = new_value
        elif param_name == 'Handling_time':
            Greedy_Algo.Handling_time = new_value
        elif param_name == 'travel_times':
            # Modify travel times matrix
            for i in range(len(Greedy_Algo.T_matrix)):
                for j in range(len(Greedy_Algo.T_matrix[i])):
                    if i != j and Greedy_Algo.T_matrix[i][j] > 0:
                        Greedy_Algo.T_matrix[i][j] = Greedy_Algo.T_matrix[i][j] * new_value
    
    def create_modified_travel_matrix(self, N, travel_multipliers):
        """Create a travel time matrix with modified values"""
        np.random.seed(42)  # For reproducibility
        base_matrix = np.random.uniform(1, 10, size=(N, N))
        
        # Zero diagonal
        for i in range(N):
            base_matrix[i][i] = 0
            
        # Apply multiplier
        modified_matrix = base_matrix * travel_multipliers
        return modified_matrix.tolist()
    
    def generate_containers_fixed(self, C, N, seed=100):
        """Generate fixed container data for reproducibility"""
        random.seed(seed)
        C_dict = {}
        
        for i in range(C):
            Dc = random.randint(24, 196)
            Oc = random.randint(Dc - 120, Dc - 24)
            P_40 = random.uniform(0.75, 0.9)
            P_Export = random.uniform(0.05, 0.7)
            
            if random.random() < P_40:
                Wc = 2
            else:
                Wc = 1
            
            if random.random() < P_Export:
                In_or_Out = 2
                Rc = random.randint(0, 24)
                Terminal = random.randint(1, N - 1)
            else:
                In_or_Out = 1
                Rc = 0
                Terminal = random.randint(1, N - 1)
            
            C_dict[i] = {
                "Rc": Rc,
                "Dc": Dc,
                "Oc": Oc,
                "Wc": Wc,
                "In_or_Out": In_or_Out,
                "Terminal": Terminal
            }
        
        return C_dict, C, N
    
    def run_single_scenario(self, scenario_name, modified_params, max_iterations=1000):
        """Run optimization for a single parameter scenario"""
        try:
            print(f"\n🔄 Running scenario: {scenario_name}")
            
            # Get scenario-specific values
            C_actual = modified_params.get('C', self.base_params['C'])
            N_actual = modified_params.get('N', self.base_params['N'])
            num_barges = modified_params.get('num_barges', self.base_params['num_barges'])
            
            # Generate fixed containers for this scenario
            C_dict, _, _ = self.generate_containers_fixed(C_actual, N_actual)
            
            # Modify Greedy_Algo parameters
            for param_name, value in modified_params.items():
                if param_name not in ['C', 'N', 'num_barges']:
                    self.modify_greedy_algo_params(param_name, value)
            
            # Set up barge parameters
            Qk = {k: modified_params.get('Qk', self.base_params['Qk']) for k in range(num_barges)}
            H_b = {k: modified_params.get('H_b', self.base_params['H_b']) for k in range(num_barges)}
            
            # Create travel matrix
            if 'travel_times' in modified_params:
                T_matrix = self.create_modified_travel_matrix(N_actual, modified_params['travel_times'])
            else:
                T_matrix = self.create_modified_travel_matrix(N_actual, 1.0)
            
            # Run greedy algorithm for initial solution
            from Greedy_Algo import get_route, check_for_cap
            
            # Get initial route assignment
            initial_assignment = {}
            for container_id in C_dict:
                route = get_route(C_dict[container_id], T_matrix)
                initial_assignment[container_id] = route
            
            # Check capacity and calculate initial cost
            _, initial_cost = check_for_cap(initial_assignment, C_dict, Qk, T_matrix)
            
            # Run meta-heuristic optimization
            optimizer = MetaHeuristic(
                C_dict=C_dict,
                Qk=Qk,
                H_b=H_b,
                T_matrix=T_matrix,
                initial_assignment=initial_assignment
            )
            
            final_assignment, final_cost = optimizer.tabu_search(max_iterations=max_iterations)
            
            # Analyze results
            barge_assignments = {}
            trucked_containers = []
            
            for container_id, route in final_assignment.items():
                if len(route) > 2:  # Has intermediate barge stops
                    barge_id = route[1]
                    if barge_id not in barge_assignments:
                        barge_assignments[barge_id] = []
                    barge_assignments[barge_id].append(container_id)
                else:
                    trucked_containers.append(container_id)
            
            # Calculate metrics
            barges_used = len(barge_assignments)
            containers_on_barges = sum(len(containers) for containers in barge_assignments.values())
            
            # Calculate utilization
            utilizations = []
            for k, containers in barge_assignments.items():
                if containers:
                    total_teu = sum(C_dict[c]['Wc'] for c in barge_assignments[k])
                    utilizations.append(total_teu / Qk[k])
            
            avg_utilization = np.mean(utilizations) if utilizations else 0
            
            result = {
                'scenario': scenario_name,
                'parameters': modified_params.copy(),
                'C_actual': C_actual,
                'N_actual': N_actual,
                'initial_cost': initial_cost,
                'final_cost': final_cost,
                'improvement': initial_cost - final_cost,
                'improvement_pct': ((initial_cost - final_cost) / initial_cost * 100) if initial_cost > 0 else 0,
                'barges_used': barges_used,
                'containers_on_barges': containers_on_barges,
                'containers_trucked': len(trucked_containers),
                'barge_utilization_pct': avg_utilization * 100,
                'success': True
            }
            
            print(f"  Final cost: €{final_cost:,.0f} (improvement: {result['improvement_pct']:.1f}%)")
            print(f"  Barges used: {barges_used}/{num_barges}, Avg utilization: {avg_utilization*100:.1f}%")
            
            # Add visual bar showing trucked vs barged container percentages
            trucked_pct = (len(trucked_containers) / C_actual) * 100
            barged_pct = (containers_on_barges / C_actual) * 100
            
            # Create a visual bar (50 characters wide)
            bar_width = 50
            trucked_chars = int((trucked_pct / 100) * bar_width)
            barged_chars = bar_width - trucked_chars
            
            # ANSI color codes for better visualization
            ORANGE = '\033[48;5;208m'  # Orange background for barged
            BLUE = '\033[48;5;21m'     # Blue background for trucked
            RESET = '\033[0m'          # Reset color
            
            # Try colored version first, fallback to simple version
            try:
                colored_bar = f"{ORANGE}{' ' * barged_chars}{BLUE}{' ' * trucked_chars}{RESET}"
                print(f"  Container allocation: |{colored_bar}|")
            except:
                # Fallback to simple characters
                bar = "█" * barged_chars + "▓" * trucked_chars
                print(f"  Container allocation: |{bar}|")
            
            print(f"  🚢 Barged: {barged_pct:.1f}% ({containers_on_barges} containers)")
            print(f"  🚛 Trucked: {trucked_pct:.1f}% ({len(trucked_containers)} containers)")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error in scenario {scenario_name}: {e}")
            return {
                'scenario': scenario_name,
                'parameters': modified_params.copy(),
                'error': str(e),
                'success': False
            }
    
    def run_parameter_sensitivity(self, param_name, base_value, variation_pcts, max_iterations=500):
        """Test parameter sensitivity with percentage variations"""
        results = []
        
        print(f"\n📊 Testing {param_name} sensitivity (base: {base_value})")
        
        for pct in variation_pcts:
            if pct == 0:
                new_value = base_value
                scenario_name = f"{param_name}_base"
            else:
                new_value = base_value * (1 + pct/100)
                scenario_name = f"{param_name}_{pct:+d}%"
            
            # Ensure non-negative values
            new_value = max(0.1, new_value)
            
            print(f"  Testing {param_name} = {new_value:.2f} ({pct:+d}%)")
            
            modified_params = self.base_params.copy()
            modified_params[param_name] = new_value
            
            result = self.run_single_scenario(scenario_name, modified_params, max_iterations)
            
            if result['success']:
                result['parameter'] = param_name
                result['variation_pct'] = pct
                result['test_value'] = new_value
                
            results.append(result)
        
        return results
    
    def run_discrete_sensitivity(self, param_name, values, max_iterations=500):
        """Test discrete parameter variations"""
        results = []
        
        print(f"\n🔢 Testing {param_name} discrete values: {values}")
        
        for value in values:
            scenario_name = f"{param_name}_{value}"
            
            print(f"  Testing {param_name} = {value}")
            
            modified_params = self.base_params.copy()
            modified_params[param_name] = value
            
            result = self.run_single_scenario(scenario_name, modified_params, max_iterations)
            
            if result['success']:
                result['parameter'] = param_name
                result['discrete_value'] = value
                
            results.append(result)
        
        return results
    
    def run_full_sensitivity_analysis(self, max_iterations=500):
        """Run complete sensitivity analysis"""
        print("🚀 Starting Full Sensitivity Analysis")
        print("="*60)
        
        all_results = []
        
        # Test continuous parameters
        continuous_params = ['Qk', 'H_b', 'H_t_40', 'H_t_20', 'Handling_time', 'travel_times']
        
        for param_name in continuous_params:
            base_value = self.base_params[param_name]
            results = self.run_parameter_sensitivity(param_name, base_value, self.variation_percentages, max_iterations)
            all_results.extend(results)
        
        # Test discrete parameters  
        for param_name, values in self.discrete_variations.items():
            results = self.run_discrete_sensitivity(param_name, values, max_iterations)
            all_results.extend(results)
        
        self.results = all_results
        return all_results
    
    def save_results(self, filename=None):
        """Save results to CSV and JSON files"""
        if not self.results:
            print("No results to save")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sensitivity_analysis_{timestamp}"
        
        # Save to CSV
        successful_results = [r for r in self.results if r.get('success', False)]
        if successful_results:
            csv_file = f"{filename}.csv"
            
            # Write CSV manually without pandas
            with open(csv_file, 'w') as f:
                if successful_results:
                    # Write header
                    headers = successful_results[0].keys()
                    f.write(','.join(headers) + '\n')
                    
                    # Write data
                    for result in successful_results:
                        values = [str(result.get(h, '')) for h in headers]
                        f.write(','.join(values) + '\n')
                        
            print(f"📁 Results saved to {csv_file}")
        
        # Save to JSON (includes failed attempts)
        json_file = f"{filename}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📁 Full results saved to {json_file}")
        
        return csv_file
    
    def create_visualizations(self, filename=None):
        """Create simplified visualizations without pandas"""
        if not self.results:
            print("No results to visualize")
            return
            
        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            print("No successful results to visualize")
            return
            
        print(f"\n📈 Creating visualizations for {len(successful_results)} scenarios...")
        
        # Group data by parameter
        param_data = {}
        for result in successful_results:
            param = result.get('parameter', 'unknown')
            if param not in param_data:
                param_data[param] = []
            param_data[param].append(result)
        
        # Figure 1: Basic Parameter Analysis
        plt.figure(figsize=(12, 8))
        plt.suptitle('Sensitivity Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Cost vs Parameters
        plt.subplot(2, 2, 1)
        for param, data in param_data.items():
            costs = [r['final_cost'] for r in data]
            variations = [r.get('variation_pct', r.get('discrete_value', 0)) for r in data]
            plt.plot(variations, costs, 'o-', label=param, alpha=0.7)
        plt.xlabel('Parameter Variation')
        plt.ylabel('Final Cost (€)')
        plt.title('Cost vs Parameter Changes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Improvement vs Parameters
        plt.subplot(2, 2, 2)
        for param, data in param_data.items():
            improvements = [r['improvement_pct'] for r in data]
            variations = [r.get('variation_pct', r.get('discrete_value', 0)) for r in data]
            plt.plot(variations, improvements, 's-', label=param, alpha=0.7)
        plt.xlabel('Parameter Variation')
        plt.ylabel('Improvement (%)')
        plt.title('Optimization Improvement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Utilization vs Parameters
        plt.subplot(2, 2, 3)
        for param, data in param_data.items():
            utilizations = [r['barge_utilization_pct'] for r in data]
            variations = [r.get('variation_pct', r.get('discrete_value', 0)) for r in data]
            plt.plot(variations, utilizations, '^-', label=param, alpha=0.7)
        plt.xlabel('Parameter Variation')
        plt.ylabel('Barge Utilization (%)')
        plt.title('Utilization vs Parameters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Containers on Barges
        plt.subplot(2, 2, 4)
        for param, data in param_data.items():
            barge_pcts = [(r['containers_on_barges'] / r['C_actual'] * 100) for r in data]
            variations = [r.get('variation_pct', r.get('discrete_value', 0)) for r in data]
            plt.plot(variations, barge_pcts, 'D-', label=param, alpha=0.7)
        plt.xlabel('Parameter Variation')
        plt.ylabel('Containers on Barges (%)')
        plt.title('Barge Allocation vs Parameters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: 3D Analysis
        self._create_simple_3d_plot(successful_results)
        
        print("✅ Visualizations created successfully!")
    
    def plot_variable_comparison(self, var1, var2, var3=None, plot_type='2d', 
                                title_prefix="Parameter Analysis", save_path=None):
        """
        Plot comparison of any variables from results data using templates
        
        Parameters:
        -----------
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
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            print("No successful results to plot")
            return None, None
            
        return self.plotter.plot_parameter_comparison(
            successful_results, var1, var2, var3, plot_type, title_prefix
        )
    
    def plot_2d_custom(self, var1, var2, **kwargs):
        """
        Create a custom 2D plot of two variables
        
        Parameters:
        -----------
        var1 : str
            X-axis variable name
        var2 : str
            Y-axis variable name
        **kwargs : dict
            Additional arguments passed to plotting template
            
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            print("No successful results to plot")
            return None, None
        
        # Extract data
        x_data = [r.get(var1, 0) for r in successful_results]
        y_data = [r.get(var2, 0) for r in successful_results]
        
        # Set default labels if not provided
        kwargs.setdefault('x_label', var1.replace('_', ' ').title())
        kwargs.setdefault('y_label', var2.replace('_', ' ').title())
        kwargs.setdefault('title', f'{var2} vs {var1}')
        
        return self.plotter.plot_2d(x_data, y_data, **kwargs)
    
    def plot_3d_custom(self, var1, var2, var3, color_var=None, **kwargs):
        """
        Create a custom 3D plot of three variables
        
        Parameters:
        -----------
        var1 : str
            X-axis variable name
        var2 : str
            Y-axis variable name  
        var3 : str
            Z-axis variable name
        color_var : str, optional
            Variable for color mapping
        **kwargs : dict
            Additional arguments passed to plotting template
            
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            print("No successful results to plot")
            return None, None
        
        # Extract data
        x_data = [r.get(var1, 0) for r in successful_results]
        y_data = [r.get(var2, 0) for r in successful_results]
        z_data = [r.get(var3, 0) for r in successful_results]
        
        color_data = None
        if color_var:
            if color_var == 'parameter':
                # Special handling for parameter names
                unique_params = list(set(r.get('parameter', 'unknown') for r in successful_results))
                param_to_num = {param: i for i, param in enumerate(unique_params)}
                color_data = [param_to_num.get(r.get('parameter', 'unknown'), 0) for r in successful_results]
            else:
                color_data = [r.get(color_var, 0) for r in successful_results]
        
        # Set default labels if not provided
        kwargs.setdefault('x_label', var1.replace('_', ' ').title())
        kwargs.setdefault('y_label', var2.replace('_', ' ').title())
        kwargs.setdefault('z_label', var3.replace('_', ' ').title())
        kwargs.setdefault('title', f'{var1} vs {var2} vs {var3}')
        
        if color_var:
            kwargs.setdefault('colorbar_label', color_var.replace('_', ' ').title())
        
        return self.plotter.plot_3d(x_data, y_data, z_data, color_data=color_data, **kwargs)
    
    def show_plots(self):
        """Display all created plots"""
        self.plotter.show_all_plots()
    
    def close_plots(self):
        """Close all plots"""
        self.plotter.close_all_plots()
    
    def get_available_variables(self):
        """
        Get list of available variables for plotting
        
        Returns:
        --------
        list : Available variable names from results
        """
        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            return []
        
        # Get all keys from the first successful result
        variables = list(successful_results[0].keys())
        
        # Filter out non-numeric variables for plotting
        numeric_vars = []
        for var in variables:
            try:
                # Test if the variable contains numeric data
                test_val = successful_results[0][var]
                if isinstance(test_val, (int, float)) or (isinstance(test_val, str) and test_val.replace('.', '').replace('-', '').isdigit()):
                    numeric_vars.append(var)
            except:
                continue
                
        return sorted(numeric_vars)

    def run_quick_analysis(self, max_iterations=150):
        """Run a quick analysis with key parameters"""
        patch_greedy_algo()
        
        print("🔍 Starting Quick Sensitivity Analysis")
        print("="*60)
        
        results = []
        
        # Test key parameters with fewer variations
        key_params = {
            'H_t_40': self.base_params['H_t_40'],
            'H_t_20': self.base_params['H_t_20'],
            'Qk': self.base_params['Qk']
        }
        
        for param_name, base_value in key_params.items():
            results.extend(self.run_parameter_sensitivity(
                param_name, base_value, [0, 20, 50], max_iterations
            ))
        
        # Test key discrete parameters
        for param_name in ['num_barges', 'C']:
            values = self.discrete_variations[param_name][:3]  # First 3 values
            results.extend(self.run_discrete_sensitivity(param_name, values, max_iterations))
        
        self.results = results
        return results
    
    def run_comprehensive_analysis(self, max_iterations=300):
        """Run the full comprehensive analysis"""
        patch_greedy_algo()
        
        print("🚀 Starting Comprehensive Sensitivity Analysis")
        print("="*60)
        print(f"  • Parameter variations: {self.variation_percentages}")
        print(f"  • Discrete variations: {self.discrete_variations}")
        print(f"  • Iterations per scenario: {max_iterations}")
        print(f"  • Estimated time: 30-60 minutes")
        print("="*60)
        
        return self.run_full_sensitivity_analysis(max_iterations)
    
    def _create_simple_3d_plot(self, results):
        """Create a simple 3D plot showing cost vs allocation vs utilization"""
        if not results:
            return
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        try:
            x = [r['final_cost'] for r in results]
            y = [(r['containers_on_barges'] / r['C_actual'] * 100) for r in results]
            z = [r['barge_utilization_pct'] for r in results]
            colors = [r['barges_used'] for r in results]
            
            scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=60, alpha=0.7)
            
            ax.set_xlabel('Final Cost (€)')
            ax.set_ylabel('Barge Allocation %')
            ax.set_zlabel('Barge Utilization %')
            ax.set_title('3D Analysis: Cost vs Allocation vs Utilization')
            
            plt.colorbar(scatter, label='Barges Used', shrink=0.5)
            plt.show()
            
        except Exception as e:
            print(f"3D plot error: {e}")
            pass

# Modify Greedy_Algo to accept fixed C and N
def patch_greedy_algo():
    """Patch the Greedy_Algo module to accept fixed C and N values"""
    import Greedy_Algo
    original_container_info = Greedy_Algo.container_info
    
    def container_info_fixed(seed, reduced, fixed_C=None, fixed_N=None):
        import random
        random.seed(seed)
        
        if fixed_C is not None and fixed_N is not None:
            C = fixed_C
            N = fixed_N
        else:
            return original_container_info(seed, reduced)
        
        C_dict = {}
        
        for i in range(C):
            Dc = random.randint(24, 196)
            Oc = random.randint(Dc - 120, Dc - 24)
            P_40 = random.uniform(0.75, 0.9)
            P_Export = random.uniform(0.05, 0.7)
            
            if random.random() < P_40:
                Wc = 2
            else:
                Wc = 1
            
            if random.random() < P_Export:
                In_or_Out = 2
                Rc = random.randint(0, 24)
                Terminal = random.randint(1, N - 1)
            else:
                In_or_Out = 1
                Rc = 0
                Terminal = random.randint(1, N - 1)
            
            C_dict[i] = {
                "Rc": Rc,
                "Dc": Dc,
                "Oc": Oc,
                "Wc": Wc,
                "In_or_Out": In_or_Out,
                "Terminal": Terminal
            }
        
        return C_dict, C, N
    
    Greedy_Algo.container_info = container_info_fixed

if __name__ == "__main__":
    analyzer = SensitivityAnalyzer()
    
    # Choose analysis type
    analysis_type = input("Choose analysis type:\n1. Quick (15 scenarios, ~5 min)\n2. Comprehensive (50+ scenarios, ~45 min)\nEnter choice (1 or 2): ").strip()
    
    if analysis_type == "1":
        results = analyzer.run_quick_analysis()
        filename = analyzer.save_results("quick_sensitivity")
    else:
        results = analyzer.run_comprehensive_analysis()
        filename = analyzer.save_results("comprehensive_sensitivity")
    
    # Create all visualizations including 3D plots
    analyzer.create_visualizations(filename.replace('.csv', '').replace('.json', ''))
    
    # Print summary
    successful = [r for r in results if r['success']]
    print(f"\n{'='*80}")
    print("🎯 ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"📊 Total scenarios: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(results) - len(successful)}")
    
    if successful:
        costs = [r['final_cost'] for r in successful]
        best = min(successful, key=lambda x: x['final_cost'])
        print(f"💰 Cost range: €{min(costs):,.0f} - €{max(costs):,.0f}")
        print(f"🏆 Best scenario: {best['scenario']} (€{best['final_cost']:,.0f})")
        print(f"📈 Avg improvement: {sum(r['improvement_pct'] for r in successful)/len(successful):.1f}%")
    
    print(f"\n📁 Files created:")
    print(f"  • {filename} (data)")
    print(f"  • Multiple visualization windows with 3D plots")
    print(f"\n🎉 Analysis complete! Check the generated plots for insights.")