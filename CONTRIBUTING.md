Contributions are always welcome. Before creating [Pull Requests](https://github.com/52North/WeatherRoutingTool/pulls) or commenting [Issues](https://github.com/52North/WeatherRoutingTool/issues) please carefully read the [Contributing](https://52north.github.io/WeatherRoutingTool/source/guidelines/contribution_guidelines.html) section in our documentation.

Please do not ask if you can work on an issue. Just re-read our documentation and remember that contributions are welcome! Also be aware that we do not assign issues to contributors we have not worked with yet. If this applies to you please do not ask to be assigned. 

## Running tests

The test suite uses pytest markers to separate fast unit tests from broader integration tests and optional/manual tests.

### Unit tests
```bash
./.venv/bin/pytest -m "unit" -q
```

### Integration tests
```bash
./.venv/bin/pytest -m "integration and not manual and not maripower" -q
```

### Manual tests
```bash
./.venv/bin/pytest -m "manual" -q
```

### Maripower tests
```bash
./.venv/bin/pytest -m "maripower" -q
```
