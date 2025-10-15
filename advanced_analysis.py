#!/usr/bin/env python3
"""
Advanced Analysis and Visualization of Sensitivity Results

This script creates detailed visualizations and analysis based on the sensitivity analysis results
to support strategic decision-making for barge operators.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import json

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_results():
    """Load the sensitivity analysis results"""
    try:
        df = pd.read_csv('full_sensitivity.csv')
        return df
    except FileNotFoundError:
        print("Results file not found. Please run the sensitivity analysis first.")
        return None

def create_cost_impact_analysis(df):
    """Create visualizations showing cost impact of different parameters"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Parameter Impact Analysis on Total Costs', fontsize=16, fontweight='bold')
    
    # 1. Truck cost parameters
    ax1 = axes[0, 0]
    h_t_40_data = df[df['parameter'] == 'H_t_40'].sort_values('variation_pct')
    h_t_20_data = df[df['parameter'] == 'H_t_20'].sort_values('variation_pct')
    
    ax1.plot(h_t_40_data['variation_pct'], h_t_40_data['final_cost'], 
             'o-', linewidth=3, markersize=8, label='40ft Truck Cost (H_t_40)', color='red')
    ax1.plot(h_t_20_data['variation_pct'], h_t_20_data['final_cost'], 
             's-', linewidth=3, markersize=8, label='20ft Truck Cost (H_t_20)', color='blue')
    
    ax1.set_xlabel('Parameter Variation (%)')
    ax1.set_ylabel('Final Cost (€)')
    ax1.set_title('Truck Cost Impact\n40ft vs 20ft Container Transport')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add cost difference annotation
    h40_base = h_t_40_data[h_t_40_data['variation_pct'] == 0]['final_cost'].iloc[0]
    h40_max = h_t_40_data[h_t_40_data['variation_pct'] == 50]['final_cost'].iloc[0]
    h20_max = h_t_20_data[h_t_20_data['variation_pct'] == 50]['final_cost'].iloc[0]
    
    ax1.annotate(f'40ft impact: +€{h40_max-h40_base:,.0f}\n20ft impact: +€{h20_max-h40_base:,.0f}',
                xy=(25, (h40_base + h40_max)/2), fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 2. Barge parameters
    ax2 = axes[0, 1]
    h_b_data = df[df['parameter'] == 'H_b'].sort_values('variation_pct')
    qk_data = df[df['parameter'] == 'Qk'].sort_values('variation_pct')
    
    ax2.plot(h_b_data['variation_pct'], h_b_data['final_cost'], 
             '^-', linewidth=3, markersize=8, label='Barge Fixed Costs (H_b)', color='green')
    ax2.plot(qk_data['variation_pct'], qk_data['final_cost'], 
             'd-', linewidth=3, markersize=8, label='Barge Capacity (Qk)', color='orange')
    
    ax2.set_xlabel('Parameter Variation (%)')
    ax2.set_ylabel('Final Cost (€)')
    ax2.set_title('Barge Parameter Impact\nFixed Costs vs Capacity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Fleet size impact
    ax3 = axes[1, 0]
    fleet_data = df[df['scenario'].str.contains('num_barges_')].copy()
    fleet_data['discrete_value'] = fleet_data['scenario'].str.extract('num_barges_(\d+)').astype(int)
    fleet_data = fleet_data.sort_values('discrete_value')
    
    if not fleet_data.empty:
        bars = ax3.bar(fleet_data['discrete_value'], fleet_data['final_cost'], 
                      color=['red' if x != 5 else 'green' for x in fleet_data['discrete_value']], alpha=0.7)
        
        # Add utilization as secondary y-axis
        ax3_twin = ax3.twinx()
        ax3_twin.plot(fleet_data['discrete_value'], fleet_data['barge_utilization_pct'], 
                     'ko-', linewidth=2, markersize=6, label='Avg Utilization %')
        ax3_twin.set_ylabel('Barge Utilization (%)', color='black')
        ax3_twin.legend(loc='upper right')
        
        ax3.set_xlabel('Number of Barges')
        ax3.set_ylabel('Final Cost (€)')
        ax3.set_title('Fleet Size Optimization\nCost vs Utilization Trade-off')
        ax3.grid(True, alpha=0.3)
        
        # Highlight optimal fleet size
        optimal_idx = fleet_data['final_cost'].idxmin()
        optimal_size = fleet_data.loc[optimal_idx, 'discrete_value']
        ax3.annotate(f'Optimal: {optimal_size} barges', 
                    xy=(optimal_size, fleet_data.loc[optimal_idx, 'final_cost']),
                    xytext=(optimal_size + 1, fleet_data.loc[optimal_idx, 'final_cost'] + 1000),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, fontweight='bold')
    
    # 4. Terminal network impact
    ax4 = axes[1, 1]
    terminal_data = df[df['scenario'].str.contains('N_')].copy()
    terminal_data['discrete_value'] = terminal_data['scenario'].str.extract('N_(\d+)').astype(int)
    terminal_data = terminal_data.sort_values('discrete_value')
    
    if not terminal_data.empty:
        # Create scatter plot with different colors for barge usage levels
        scatter = ax4.scatter(terminal_data['discrete_value'], terminal_data['final_cost'], 
                            c=terminal_data['containers_on_barges'], s=100, 
                            cmap='RdYlBu_r', alpha=0.8, edgecolors='black')
        
        plt.colorbar(scatter, ax=ax4, label='Containers on Barges')
        
        ax4.set_xlabel('Number of Terminals')
        ax4.set_ylabel('Final Cost (€)')
        ax4.set_title('Terminal Network Impact\nCost vs Barge Utilization')
        ax4.grid(True, alpha=0.3)
        
        # Add annotations for key insights
        best_terminal = terminal_data.loc[terminal_data['final_cost'].idxmin()]
        ax4.annotate(f'Best: {best_terminal["discrete_value"]} terminals\n€{best_terminal["final_cost"]:,.0f}',
                    xy=(best_terminal['discrete_value'], best_terminal['final_cost']),
                    xytext=(best_terminal['discrete_value'] + 2, best_terminal['final_cost'] + 2000),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('cost_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_efficiency_dashboard(df):
    """Create a dashboard showing efficiency metrics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Operational Efficiency Dashboard', fontsize=18, fontweight='bold')
    
    # 1. Cost vs Container Volume
    ax1 = axes[0, 0]
    container_data = df[df['scenario'].str.contains('C_')].copy()
    container_data['discrete_value'] = container_data['scenario'].str.extract('C_(\d+)').astype(int)
    container_data = container_data.sort_values('discrete_value')
    
    if not container_data.empty:
        ax1.plot(container_data['discrete_value'], container_data['final_cost'], 'o-', linewidth=3, markersize=10, color='purple')
        ax1.set_xlabel('Number of Containers')
        ax1.set_ylabel('Final Cost (€)')
        ax1.set_title('Volume vs Cost\nEconomies of Scale Analysis')
        ax1.grid(True, alpha=0.3)
        
        # Calculate and show cost per container
        ax1_twin = ax1.twinx()
        cost_per_container = container_data['final_cost'] / container_data['discrete_value']
        ax1_twin.plot(container_data['discrete_value'], cost_per_container, 's--', color='red', label='Cost per Container')
        ax1_twin.set_ylabel('Cost per Container (€)', color='red')
        ax1_twin.legend()
    
    # 2. Barge vs Truck Allocation
    ax2 = axes[0, 1]
    # Use baseline scenarios to show allocation patterns
    baseline_scenarios = df[df['scenario'].str.contains('base|_base')].copy()
    
    if not baseline_scenarios.empty:
        categories = ['Truck Cost\nBase', 'Barge Cost\nBase', 'Handling\nBase', 'Travel\nBase']
        barge_pcts = []
        truck_pcts = []
        
        for scenario in ['H_t_40_base', 'H_b_base', 'Handling_time_base', 'travel_times_base']:
            scenario_data = baseline_scenarios[baseline_scenarios['scenario'] == scenario]
            if not scenario_data.empty:
                barge_pct = (scenario_data['containers_on_barges'].iloc[0] / scenario_data['C_actual'].iloc[0]) * 100
                truck_pct = 100 - barge_pct
                barge_pcts.append(barge_pct)
                truck_pcts.append(truck_pct)
        
        x = np.arange(len(categories))
        width = 0.6
        
        ax2.bar(x, truck_pcts, width, label='Trucked %', color='lightcoral', alpha=0.8)
        ax2.bar(x, barge_pcts, width, bottom=truck_pcts, label='Barged %', color='lightblue', alpha=0.8)
        
        ax2.set_xlabel('Parameter Categories')
        ax2.set_ylabel('Container Allocation (%)')
        ax2.set_title('Container Allocation Distribution\nBarge vs Truck')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Utilization Efficiency
    ax3 = axes[0, 2]
    # Compare utilization across different scenarios
    scenarios_with_util = df[df['barge_utilization_pct'] > 0].copy()
    
    if not scenarios_with_util.empty:
        # Group by parameter and get mean utilization
        param_utilization = scenarios_with_util.groupby('parameter')['barge_utilization_pct'].mean().sort_values(ascending=False)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(param_utilization)))
        bars = ax3.barh(range(len(param_utilization)), param_utilization.values, color=colors)
        
        ax3.set_yticks(range(len(param_utilization)))
        ax3.set_yticklabels(param_utilization.index)
        ax3.set_xlabel('Average Barge Utilization (%)')
        ax3.set_title('Barge Utilization by Parameter\nHigher is Better')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, param_utilization.values)):
            ax3.text(value + 1, i, f'{value:.1f}%', va='center', fontweight='bold')
    
    # 4. Cost Sensitivity Heatmap
    ax4 = axes[1, 0]
    # Create a heatmap of cost changes by parameter and variation level
    
    param_variations = df[df['parameter'].isin(['H_t_40', 'H_t_20', 'H_b', 'travel_times'])].copy()
    if not param_variations.empty:
        pivot_data = param_variations.pivot(index='parameter', columns='variation_pct', values='final_cost')
        
        # Calculate percentage change from baseline (0%)
        baseline_costs = pivot_data[0]
        for col in pivot_data.columns:
            if col != 0:
                pivot_data[col] = ((pivot_data[col] - baseline_costs) / baseline_costs * 100)
        
        # Remove the 0% column as it will be all zeros
        pivot_data = pivot_data.drop(columns=[0])
        
        im = ax4.imshow(pivot_data.values, cmap='RdYlBu_r', aspect='auto')
        
        ax4.set_xticks(range(len(pivot_data.columns)))
        ax4.set_xticklabels([f'+{x}%' for x in pivot_data.columns])
        ax4.set_yticks(range(len(pivot_data.index)))
        ax4.set_yticklabels(pivot_data.index)
        ax4.set_xlabel('Parameter Variation')
        ax4.set_ylabel('Parameter Type')
        ax4.set_title('Cost Sensitivity Heatmap\n% Change from Baseline')
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                value = pivot_data.iloc[i, j]
                ax4.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                        color='white' if abs(value) > 5 else 'black', fontweight='bold')
        
        plt.colorbar(im, ax=ax4, label='Cost Change (%)')
    
    # 5. Optimal Operating Regions
    ax5 = axes[1, 1]
    
    # Create a scatter plot showing the relationship between different metrics
    all_successful = df[df['success'] == True].copy()
    
    if not all_successful.empty:
        scatter = ax5.scatter(all_successful['containers_on_barges'], all_successful['final_cost'],
                            c=all_successful['barge_utilization_pct'], s=60, alpha=0.7, cmap='viridis')
        
        ax5.set_xlabel('Containers on Barges')
        ax5.set_ylabel('Final Cost (€)')
        ax5.set_title('Operating Efficiency Map\nCost vs Allocation vs Utilization')
        plt.colorbar(scatter, ax=ax5, label='Barge Utilization (%)')
        ax5.grid(True, alpha=0.3)
        
        # Highlight the best scenarios
        best_scenarios = all_successful.nsmallest(5, 'final_cost')
        ax5.scatter(best_scenarios['containers_on_barges'], best_scenarios['final_cost'],
                   s=100, facecolors='none', edgecolors='red', linewidths=3, 
                   label='Top 5 Scenarios')
        ax5.legend()
    
    # 6. ROI Analysis
    ax6 = axes[1, 2]
    
    # Calculate potential savings for each parameter optimization
    savings_analysis = []
    
    for param in ['H_t_40', 'H_t_20', 'H_b', 'travel_times']:
        param_data = df[df['parameter'] == param].copy()
        if not param_data.empty:
            baseline_cost = param_data[param_data['variation_pct'] == 0]['final_cost'].iloc[0]
            max_cost = param_data['final_cost'].max()
            max_savings = max_cost - baseline_cost
            
            savings_analysis.append({
                'parameter': param,
                'max_savings': max_savings,
                'baseline_cost': baseline_cost
            })
    
    if savings_analysis:
        savings_df = pd.DataFrame(savings_analysis)
        savings_df['savings_pct'] = (savings_df['max_savings'] / savings_df['baseline_cost']) * 100
        savings_df = savings_df.sort_values('max_savings', ascending=True)
        
        bars = ax6.barh(range(len(savings_df)), savings_df['max_savings'], 
                       color=['red', 'orange', 'yellow', 'green'][:len(savings_df)])
        
        ax6.set_yticks(range(len(savings_df)))
        ax6.set_yticklabels(savings_df['parameter'])
        ax6.set_xlabel('Maximum Cost Impact (€)')
        ax6.set_title('Parameter Optimization Priority\nHighest Impact Parameters')
        ax6.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (bar, value, pct) in enumerate(zip(bars, savings_df['max_savings'], savings_df['savings_pct'])):
            ax6.text(value + 200, i, f'€{value:,.0f}\n({pct:.1f}%)', 
                    va='center', fontweight='bold', ha='left')
    
    plt.tight_layout()
    plt.savefig('efficiency_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_strategic_recommendations_chart(df):
    """Create a visual summary of strategic recommendations"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Strategic Recommendations Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Parameter Priority Matrix
    ax1.set_title('Parameter Optimization Priority Matrix', fontweight='bold')
    
    # Define impact vs effort matrix
    parameters = {
        'H_t_40 (40ft Truck)': {'impact': 9, 'effort': 2, 'color': 'red'},
        'H_b (Barge Costs)': {'impact': 7, 'effort': 5, 'color': 'orange'},
        'Network Size': {'impact': 8, 'effort': 6, 'color': 'green'},
        'Fleet Size': {'impact': 6, 'effort': 3, 'color': 'blue'},
        'H_t_20 (20ft Truck)': {'impact': 2, 'effort': 2, 'color': 'gray'},
        'Travel Times': {'impact': 4, 'effort': 7, 'color': 'purple'},
        'Barge Capacity': {'impact': 1, 'effort': 9, 'color': 'brown'},
        'Handling Time': {'impact': 1, 'effort': 4, 'color': 'pink'}
    }
    
    for param, values in parameters.items():
        ax1.scatter(values['effort'], values['impact'], s=200, c=values['color'], alpha=0.7)
        ax1.annotate(param, (values['effort'], values['impact']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Implementation Effort (1-10 scale)')
    ax1.set_ylabel('Business Impact (1-10 scale)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Add quadrant labels
    ax1.text(2, 8.5, 'Quick Wins\n(High Impact, Low Effort)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7), fontweight='bold')
    ax1.text(8, 8.5, 'Strategic Projects\n(High Impact, High Effort)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7), fontweight='bold')
    ax1.text(2, 1.5, 'Fill-ins\n(Low Impact, Low Effort)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7), fontweight='bold')
    ax1.text(8, 1.5, 'Avoid\n(Low Impact, High Effort)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7), fontweight='bold')
    
    # 2. Cost Optimization Potential
    ax2.set_title('Cost Optimization Potential by Strategy', fontweight='bold')
    
    strategies = [
        'Current Operations',
        'Optimize Truck Rates',
        'Network Optimization', 
        'Fleet Right-sizing',
        'Demand Management',
        'Combined Strategy'
    ]
    
    costs = [35016, 25016, 26900, 33000, 17808, 15000]  # Estimated based on analysis
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'gold']
    
    bars = ax2.bar(range(len(strategies)), costs, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.set_ylabel('Estimated Cost (€)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add cost labels on bars
    for bar, cost in zip(bars, costs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'€{cost:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight savings
    baseline_cost = costs[0]
    for i, cost in enumerate(costs[1:], 1):
        savings = baseline_cost - cost
        savings_pct = (savings / baseline_cost) * 100
        if savings > 0:
            ax2.text(i, cost/2, f'Save €{savings:,.0f}\n({savings_pct:.1f}%)', 
                    ha='center', va='center', fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.8))
    
    # 3. Fleet Utilization Optimization
    ax3.set_title('Fleet Size vs Utilization Analysis', fontweight='bold')
    
    fleet_data = df[df['scenario'].str.contains('num_barges_')].copy()
    fleet_data['discrete_value'] = fleet_data['scenario'].str.extract('num_barges_(\d+)').astype(int)
    fleet_data = fleet_data.sort_values('discrete_value')
    
    if not fleet_data.empty:
        # Create combined plot
        ax3_cost = ax3
        ax3_util = ax3.twinx()
        
        # Cost bars
        bars = ax3_cost.bar(fleet_data['discrete_value'], fleet_data['final_cost'], 
                           alpha=0.6, color='lightcoral', label='Total Cost')
        
        # Utilization line
        line = ax3_util.plot(fleet_data['discrete_value'], fleet_data['barge_utilization_pct'], 
                           'go-', linewidth=3, markersize=8, label='Avg Utilization %')
        
        ax3_cost.set_xlabel('Number of Barges')
        ax3_cost.set_ylabel('Total Cost (€)', color='red')
        ax3_util.set_ylabel('Average Utilization (%)', color='green')
        
        # Highlight optimal zone
        optimal_range = fleet_data[(fleet_data['discrete_value'] >= 5) & (fleet_data['discrete_value'] <= 6)]
        for _, row in optimal_range.iterrows():
            ax3_cost.bar(row['discrete_value'], row['final_cost'], 
                        alpha=0.9, color='green', width=0.6)
        
        ax3.text(5.5, max(fleet_data['final_cost']) * 0.8, 'Optimal\nZone', 
                ha='center', va='center', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # 4. Implementation Timeline
    ax4.set_title('Recommended Implementation Timeline', fontweight='bold')
    
    # Create Gantt-like chart
    phases = {
        'Phase 1\n(0-3 months)': {'start': 0, 'duration': 3, 'color': 'red'},
        'Phase 2\n(3-6 months)': {'start': 3, 'duration': 3, 'color': 'orange'}, 
        'Phase 3\n(6-12 months)': {'start': 6, 'duration': 6, 'color': 'green'}
    }
    
    activities = [
        ['Truck Rate Negotiation', 0, 1, 'red'],
        ['Demand Batching', 0, 2, 'red'],
        ['Terminal Analysis', 1, 2, 'orange'],
        ['Network Optimization', 3, 3, 'orange'],
        ['Route Systems', 4, 2, 'orange'],
        ['Strategic Positioning', 6, 6, 'green'],
        ['Advanced Forecasting', 8, 4, 'green']
    ]
    
    y_pos = 0
    for activity, start, duration, color in activities:
        ax4.barh(y_pos, duration, left=start, height=0.6, 
                color=color, alpha=0.7, edgecolor='black')
        ax4.text(start + duration/2, y_pos, activity, 
                ha='center', va='center', fontweight='bold', fontsize=9)
        y_pos += 1
    
    ax4.set_xlabel('Months from Start')
    ax4.set_ylabel('Implementation Activities')
    ax4.set_xlim(0, 12)
    ax4.set_ylim(-0.5, len(activities) - 0.5)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add phase backgrounds
    ax4.axvspan(0, 3, alpha=0.1, color='red', label='Phase 1: Quick Wins')
    ax4.axvspan(3, 6, alpha=0.1, color='orange', label='Phase 2: Operations')
    ax4.axvspan(6, 12, alpha=0.1, color='green', label='Phase 3: Strategic')
    ax4.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('strategic_recommendations.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_executive_summary_stats(df):
    """Generate key statistics for executive summary"""
    
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY STATISTICS")
    print("="*80)
    
    # Overall performance metrics
    total_scenarios = len(df)
    successful_scenarios = len(df[df['success'] == True])
    
    print(f"📊 Total Scenarios Analyzed: {total_scenarios}")
    print(f"✅ Successful Scenarios: {successful_scenarios}")
    print(f"📈 Success Rate: {successful_scenarios/total_scenarios*100:.1f}%")
    
    # Cost analysis
    successful_df = df[df['success'] == True]
    min_cost = successful_df['final_cost'].min()
    max_cost = successful_df['final_cost'].max()
    avg_cost = successful_df['final_cost'].mean()
    best_scenario = successful_df.loc[successful_df['final_cost'].idxmin(), 'scenario']
    
    print(f"\n💰 COST ANALYSIS:")
    print(f"   • Minimum Cost: €{min_cost:,.0f} ({best_scenario})")
    print(f"   • Maximum Cost: €{max_cost:,.0f}")
    print(f"   • Average Cost: €{avg_cost:,.0f}")
    print(f"   • Cost Range: {((max_cost-min_cost)/min_cost)*100:.1f}% variation")
    
    # Parameter impact ranking
    print(f"\n🎯 TOP PARAMETER IMPACTS:")
    param_impacts = {}
    
    for param in ['H_t_40', 'H_t_20', 'H_b', 'travel_times']:
        param_data = successful_df[successful_df['parameter'] == param]
        if not param_data.empty:
            baseline_cost = param_data[param_data['variation_pct'] == 0]['final_cost'].iloc[0]
            max_cost_param = param_data['final_cost'].max()
            impact = max_cost_param - baseline_cost
            param_impacts[param] = impact
    
    sorted_impacts = sorted(param_impacts.items(), key=lambda x: x[1], reverse=True)
    for i, (param, impact) in enumerate(sorted_impacts, 1):
        print(f"   {i}. {param}: +€{impact:,.0f} max impact")
    
    # Fleet optimization insights
    print(f"\n🚢 FLEET OPTIMIZATION INSIGHTS:")
    fleet_data = successful_df[successful_df['scenario'].str.contains('num_barges_')].copy()
    if not fleet_data.empty:
        fleet_data['discrete_value'] = fleet_data['scenario'].str.extract('num_barges_(\d+)').astype(int)
        optimal_fleet = fleet_data.loc[fleet_data['final_cost'].idxmin()]
        print(f"   • Optimal Fleet Size: {optimal_fleet['discrete_value']:.0f} barges")
        print(f"   • Optimal Cost: €{optimal_fleet['final_cost']:,.0f}")
        print(f"   • Optimal Utilization: {optimal_fleet['barge_utilization_pct']:.1f}%")
    
    # Network insights
    print(f"\n🌐 NETWORK OPTIMIZATION INSIGHTS:")
    network_data = successful_df[successful_df['scenario'].str.contains('N_')].copy()
    if not network_data.empty:
        network_data['discrete_value'] = network_data['scenario'].str.extract('N_(\d+)').astype(int)
        best_network = network_data.loc[network_data['final_cost'].idxmin()]
        worst_network = network_data.loc[network_data['final_cost'].idxmax()]
        print(f"   • Best Network Size: {best_network['discrete_value']:.0f} terminals (€{best_network['final_cost']:,.0f})")
        print(f"   • Worst Network Size: {worst_network['discrete_value']:.0f} terminals (€{worst_network['final_cost']:,.0f})")
        print(f"   • Network Impact: €{worst_network['final_cost'] - best_network['final_cost']:,.0f} difference")
    
    # Volume insights
    print(f"\n📦 VOLUME OPTIMIZATION INSIGHTS:")
    volume_data = successful_df[successful_df['scenario'].str.contains('C_')].copy()
    if not volume_data.empty:
        volume_data['discrete_value'] = volume_data['scenario'].str.extract('C_(\d+)').astype(int)
        best_volume = volume_data.loc[volume_data['final_cost'].idxmin()]
        volume_data['cost_per_container'] = volume_data['final_cost'] / volume_data['discrete_value']
        most_efficient = volume_data.loc[volume_data['cost_per_container'].idxmin()]
        print(f"   • Best Volume: {best_volume['discrete_value']:.0f} containers (€{best_volume['final_cost']:,.0f})")
        print(f"   • Most Efficient: {most_efficient['discrete_value']:.0f} containers (€{most_efficient['cost_per_container']:.0f}/container)")
    
    print("\n" + "="*80)

def main():
    """Main function to run all analyses"""
    
    print("Loading sensitivity analysis results...")
    df = load_results()
    
    if df is None:
        return
    
    print("Creating cost impact analysis...")
    create_cost_impact_analysis(df)
    
    print("Creating efficiency dashboard...")
    create_efficiency_dashboard(df)
    
    print("Creating strategic recommendations chart...")
    create_strategic_recommendations_chart(df)
    
    print("Generating executive summary statistics...")
    generate_executive_summary_stats(df)
    
    print("\n🎉 Analysis complete! Check the generated visualizations:")
    print("   • cost_impact_analysis.png")
    print("   • efficiency_dashboard.png") 
    print("   • strategic_recommendations.png")
    print("\n📋 Full report available in: Sensitivity_Analysis_Report.md")

if __name__ == "__main__":
    main()