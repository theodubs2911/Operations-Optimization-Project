# Sensitivity Analysis for Operations Optimization Project

## Overview
This streamlined sensitivity analysis system evaluates how parameter changes affect the container allocation optimization problem. The system has been cleaned up to provide comprehensive analysis with 3D visualizations while maintaining all functionality.

## Key Features

### ✅ **Simplified Structure**
- Single file: `sensitivity_analysis.py` (all others removed)
- Interactive mode: Choose Quick (15 scenarios) or Comprehensive (50+ scenarios)
- Streamlined parameter ranges for faster execution

### 📊 **Analysis Capabilities**
- **Continuous Parameters**: H_t_40, H_t_20, Handling_time, Qk, H_b, travel_times
- **Discrete Parameters**: num_barges, N (terminals), C (containers)
- **Variation Range**: -20%, -10%, 0%, +10%, +20%
- **Container Allocation Visualization**: Visual bars showing barged vs trucked percentages

### 🎯 **3D Visualizations**
- **3D Surface Plots**: Two parameters vs cost
- **3D Scatter Plots**: Container count vs terminals vs allocation percentage  
- **3D Parameter Plots**: Barge capacity vs fixed cost vs utilization
- **3D Cost Analysis**: Cost vs allocation % vs utilization %
- **3D Efficiency Analysis**: Cost per container vs allocation vs utilization
- **3D Parameter Interactions**: Multi-dimensional parameter effects

### 📈 **Output Files**
- **Data**: CSV and JSON formats with detailed results
- **Visualizations**: 5 separate plot files (PNG + PDF):
  1. `*_continuous_analysis` - Parameter sensitivity curves
  2. `*_discrete_analysis` - Bar charts for discrete parameters
  3. `*_advanced_analysis` - Heatmaps and scatter plots
  4. `*_allocation_visualization` - Container allocation breakdowns
  5. `*_3d_analysis` - 3D plots comparing dual variables

## Usage

### Quick Analysis (5 minutes)
```bash
python sensitivity_analysis.py
# Choose option 1
```
- Tests 15 key scenarios
- Parameters: H_t_40, H_t_20, Qk, num_barges, C
- 150 iterations per scenario

### Comprehensive Analysis (45 minutes)  
```bash
python sensitivity_analysis.py
# Choose option 2
```
- Tests 50+ scenarios
- All parameters with full variation ranges
- 300 iterations per scenario

## Key Insights from Analysis

### 🔍 **Parameter Sensitivity**
- **Container Count (C)**: Most impactful - affects both absolute cost and allocation strategy
- **Truck Costs (H_t_40, H_t_20)**: Direct linear impact on total cost
- **Number of Barges**: More barges = slightly better allocation but higher fixed costs

### 📊 **Allocation Patterns**
- Most scenarios are truck-heavy (90%+ trucked containers)
- Barge utilization typically low (5-15% on average)
- Sweet spot appears around 100-150 containers for better barge efficiency

### 💡 **3D Analysis Reveals**
- Complex interactions between container count and terminal count
- Non-linear relationships between barge capacity and utilization
- Cost efficiency improvements possible with balanced parameter tuning

## Visual Features

### 🎨 **Enhanced Terminal Output**
- Color-coded allocation bars (orange = barged, blue = trucked)  
- Progress indicators with percentages
- Emoji-enhanced summary statistics
- Container allocation breakdown for each scenario

### 📊 **3D Plot Types**
- **Surface plots** for continuous parameter interactions
- **Scatter plots** for discrete parameter combinations
- **Multi-dimensional visualizations** showing cost, allocation, and efficiency
- **Parameter interaction plots** revealing complex relationships

## File Structure
```
sensitivity_analysis.py          # Main analysis engine (only file needed)
├── Quick mode: 15 scenarios
├── Comprehensive mode: 50+ scenarios  
├── 3D plotting capabilities
├── Multiple visualization outputs
└── Detailed CSV/JSON results
```

## Benefits of Streamlined System

✅ **Simplified**: Single file, no extra dependencies  
✅ **Comprehensive**: All original functionality preserved  
✅ **Visual**: Enhanced 3D plots and terminal output  
✅ **Flexible**: Interactive mode selection  
✅ **Professional**: Publication-ready visualizations  
✅ **Insightful**: 3D analysis reveals parameter interactions  

## Example Results
- **Best Scenario**: C_100 (€29,748) - fewer containers, better efficiency
- **Worst Scenario**: Typically high container counts with expensive trucking
- **Key Finding**: Container count and truck costs are most sensitive parameters
- **3D Insight**: Non-linear interactions between barge capacity and terminal count

Run the analysis to explore how parameter changes affect your optimization results!