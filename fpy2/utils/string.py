import re

"""
Operations on strings.
"""

def pythonize_id(name: str) -> str:
    """
    Converts a string to a valid Python identifier.

    Any dash or space is replaced with an underscore, and any invalid
    characters are replaced with its Unicode representation.
    """

    # Replace hyphens with underscores
    name = name.replace('-', '_').replace(' ', '_')

    # Replace invalid characters
    name = re.sub(r'\W|^(?=\d)', lambda x: f'_u{ord(x.group(0))}_', name)

    return name
