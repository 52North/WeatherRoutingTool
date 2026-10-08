# Installation Guide

## Dependency Management

WeatherRoutingTool uses optional dependency groups to provide a flexible installation experience. This allows users to install only the dependencies they need, reducing installation time and potential conflicts.

## Installation Options

### 1. Minimal Installation (Recommended for beginners)
```bash
pip install WeatherRoutingTool
```

**What you get:**
- Core weather routing functionality
- Basic data processing capabilities
- Configuration management
- Essential scientific computing tools

**Use case:** Learning weather routing, basic route optimization, or when you don't need visualization.

### 2. With Visualization
```bash
pip install WeatherRoutingTool[viz]
```

**What you get:**
- Everything from minimal installation
- Route visualization and plotting
- Map generation
- Statistical plots

**Use case:** When you need to visualize routes, weather conditions, or optimization results.

### 3. With Geospatial Features
```bash
pip install WeatherRoutingTool[geospatial]
```

**What you get:**
- Everything from minimal installation
- Advanced geographic calculations
- Land mask support
- Geospatial data analysis

**Use case:** When working with complex geographic constraints or advanced route planning.

### 4. With Genetic Algorithm
```bash
pip install WeatherRoutingTool[genetic]
```

**What you get:**
- Everything from minimal installation
- Genetic algorithm optimization
- Multi-objective optimization capabilities

**Use case:** When you need advanced optimization algorithms beyond basic routing.

### 5. With Data Processing
```bash
pip install WeatherRoutingTool[data]
```

**What you get:**
- Everything from minimal installation
- Large dataset processing
- Parallel computing capabilities
- Advanced data access

**Use case:** When working with large weather datasets or need performance optimization.

### 6. With External Data Download
```bash
pip install WeatherRoutingTool[download]
```

**What you get:**
- Everything from minimal installation
- Automatic weather data downloading
- Access to external data sources

**Use case:** When you need to fetch weather data from external sources automatically.

### 7. Full Installation
```bash
pip install WeatherRoutingTool[all]
```

**What you get:** All features and dependencies.

**Use case:** When you need all features or are unsure which groups you need.

### 8. Development Installation
```bash
pip install WeatherRoutingTool[dev]
```

**What you get:**
- All features
- Development tools (flake8, pytest)
- Testing framework

**Use case:** For contributors and developers working on the codebase.

## Combining Groups

You can combine multiple groups:
```bash
pip install WeatherRoutingTool[viz,genetic]
```

## Upgrading Installations

To add features to an existing installation:
```bash
pip install WeatherRoutingTool[viz]  # Adds visualization to existing installation
```

## Troubleshooting

### Installation Issues

1. **Cartopy installation fails on Windows:**
   ```bash
   pip install WeatherRoutingTool[viz] --no-binary cartopy
   ```

2. **Memory errors during installation:**
   Use minimal installation and add groups as needed:
   ```bash
   pip install WeatherRoutingTool
   pip install WeatherRoutingTool[viz]
   ```

3. **Conflicts with existing packages:**
   Use a virtual environment:
   ```bash
   python -m venv wrt_env
   source wrt_env/bin/activate  # On Windows: wrt_env\Scripts\activate
   pip install WeatherRoutingTool[all]
   ```

### Dependency Conflicts

If you encounter dependency conflicts, try:
1. Using a fresh virtual environment
2. Installing groups individually to identify the conflicting package
3. Using `pip install --upgrade` to update conflicting packages

## Performance Considerations

- **Minimal installation** has the fastest installation time and smallest footprint
- **Full installation** provides all features but takes longer to install
- **Selective installation** reduces the chance of dependency conflicts
- **Virtual environments** are recommended for all installations

## Feature Requirements

| Feature | Required Dependency Group |
|---------|---------------------------|
| Basic routing | core (included in all installations) |
| Route plotting | viz |
| Map generation | viz |
| Genetic algorithm | genetic |
| Land constraints | geospatial |
| Large datasets | data |
| Auto-download weather | download |
| Image processing | image |

## Migration from Previous Versions

If you're upgrading from a version before dependency groups:
1. Your existing installation will continue to work
2. To use the new modular approach, create a fresh installation:
   ```bash
   pip uninstall WeatherRoutingTool
   pip install WeatherRoutingTool[all]  # Or your preferred groups
   ```
