"""Example module for the template."""


def hello_world(name: str = "World") -> str:
    """Return a greeting message.

    Args:
        name: The name to greet. Defaults to "World".

    Returns:
        A greeting message.

    Example:
        >>> hello_world("Python")
        'Hello, Python!'
    """
    return f"Hello, {name}!"


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b

    Example:
        >>> add_numbers(2, 3)
        5
    """
    return a + b
