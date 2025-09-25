"""Tests for the example module."""

import pytest

from src.example import add_numbers, hello_world


class TestHelloWorld:
    """Test cases for hello_world function."""

    def test_hello_world_default(self):
        """Test hello_world with default parameter."""
        result = hello_world()
        assert result == "Hello, World!"

    def test_hello_world_with_name(self):
        """Test hello_world with custom name."""
        result = hello_world("Python")
        assert result == "Hello, Python!"

    def test_hello_world_empty_string(self):
        """Test hello_world with empty string."""
        result = hello_world("")
        assert result == "Hello, !"

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Alice", "Hello, Alice!"),
            ("Bob", "Hello, Bob!"),
            ("123", "Hello, 123!"),
        ],
    )
    def test_hello_world_parametrized(self, name, expected):
        """Test hello_world with various inputs."""
        assert hello_world(name) == expected


class TestAddNumbers:
    """Test cases for add_numbers function."""

    def test_add_positive_numbers(self):
        """Test adding positive numbers."""
        result = add_numbers(2, 3)
        assert result == 5

    def test_add_negative_numbers(self):
        """Test adding negative numbers."""
        result = add_numbers(-2, -3)
        assert result == -5

    def test_add_mixed_numbers(self):
        """Test adding positive and negative numbers."""
        result = add_numbers(5, -3)
        assert result == 2

    def test_add_zero(self):
        """Test adding zero."""
        result = add_numbers(5, 0)
        assert result == 5

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (1, 1, 2),
            (10, 20, 30),
            (-5, 5, 0),
            (100, -50, 50),
        ],
    )
    def test_add_numbers_parametrized(self, a, b, expected):
        """Test add_numbers with various inputs."""
        assert add_numbers(a, b) == expected


# Integration tests
class TestIntegration:
    """Integration test cases."""

    def test_hello_world_and_add_numbers(self):
        """Test using both functions together."""
        name = "Python"
        greeting = hello_world(name)
        number_sum = add_numbers(len(name), 10)

        assert greeting == "Hello, Python!"
        assert number_sum == 16  # len("Python") + 10 = 6 + 10 = 16


# Slow tests (can be skipped with -m "not slow")
@pytest.mark.slow
class TestSlowOperations:
    """Slow test cases."""

    def test_large_number_addition(self):
        """Test adding very large numbers."""
        large_num = 10**10
        result = add_numbers(large_num, large_num)
        assert result == 2 * large_num
