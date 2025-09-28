# BMS Agent Testing Strategy

## Overview
This document outlines the testing strategy for the BMS Agent application, ensuring reliability, performance, and maintainability of the system.

## Test Types

### 1. Unit Tests
- **Purpose**: Test individual components in isolation
- **Location**: `tests/unit/`
- **Coverage**: 
  - Core business logic
  - Utility functions
  - Data transformations
- **Tools**: `pytest`, `pytest-mock`

### 2. Integration Tests
- **Purpose**: Test interactions between components
- **Location**: `tests/integration/`
- **Coverage**:
  - API endpoints
  - Database operations
  - External service integrations
- **Tools**: `pytest`, `httpx`

### 3. End-to-End Tests
- **Purpose**: Test complete workflows
- **Location**: `tests/e2e/`
- **Coverage**:
  - User journeys
  - Complete document processing flow
  - Search functionality
- **Tools**: `pytest`, `playwright`

## Test Environment

### Dependencies
- Python 3.11+
- Qdrant database
- Ollama service
- Test API key

### Setup
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Set environment variables
export BMS_API_KEY=test-key-123
export QDRANT_URL=http://localhost:6333
```

## Running Tests

### Run All Tests
```bash
pytest -v --cov=./ --cov-report=term-missing
```

### Run Specific Test Type
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# E2E tests
pytest tests/e2e/
```

### Generate Coverage Report
```bash
pytest --cov=./ --cov-report=html
# Open htmlcov/index.html in browser
```

## CI/CD Pipeline

The CI/CD pipeline runs on every push and pull request:

1. **Test Job**:
   - Sets up Python environment
   - Installs dependencies
   - Runs all tests with coverage
   - Uploads coverage to Codecov

2. **Deploy Job** (main branch only):
   - Runs after successful tests
   - Deploys to production environment

## Test Data Management

### Fixtures
- Located in `tests/fixtures/`
- Include sample documents and expected outputs
- Used for consistent test data

### Mocks
- Used for external services
- Implemented using `unittest.mock`
- Located in respective test files

## Performance Testing

### Load Testing
- **Tool**: `locust`
- **Location**: `tests/performance/`
- **Scenarios**:
  - Concurrent document uploads
  - High-volume search requests

### Benchmarking
- **Tool**: `pytest-benchmark`
- **Metrics**:
  - Response times
  - Throughput
  - Resource usage

## Security Testing

### Static Analysis
- **Tool**: `bandit`
- **Command**: `bandit -r .`

### Dependency Scanning
- **Tool**: `safety`
- **Command**: `safety check`

## Test Maintenance

### Adding New Tests
1. Identify test type (unit/integration/e2e)
2. Create test file in appropriate directory
3. Follow existing patterns
4. Include proper assertions
5. Add necessary fixtures/mocks

### Updating Tests
- Update tests when corresponding code changes
- Ensure all tests pass before merging
- Update fixtures if data structure changes

## Monitoring and Reporting

### Test Results
- HTML reports in `htmlcov/`
- Console output with detailed failures
- Codecov integration for PRs

### Metrics
- Code coverage percentage
- Test execution time
- Flaky test detection

## Best Practices

1. **Naming Conventions**:
   - Test files: `test_*.py`
   - Test functions: `test_*`
   - Test classes: `Test*`

2. **Test Isolation**:
   - Each test should be independent
   - Use fixtures for setup/teardown
   - Clean up test data after tests

3. **Assertions**:
   - One logical assertion per test
   - Use descriptive assertion messages
   - Test both happy path and error cases

4. **Documentation**:
   - Document test purpose
   - Include example inputs/outputs
   - Note any test dependencies
