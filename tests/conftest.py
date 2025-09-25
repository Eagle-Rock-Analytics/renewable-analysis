"""Test configuration and fixtures."""

import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"name": "test", "value": 42}


@pytest.fixture
def sample_list():
    """Provide a sample list for tests."""
    return [1, 2, 3, 4, 5]
