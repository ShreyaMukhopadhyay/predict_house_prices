def format_y_tick(value, currency: str = False):
    """
    Format a given numerical value into a human-readable string with appropriate suffix (B for billion,
    M for million, K for thousand), optionally prefixed by a specified currency symbol.

    Parameters:
    - value: int or float
        The numerical value to format.
    - currency: str, optional
        A string representing the currency symbol to prefix the formatted number. Defaults to False (no prefix).

    Returns:
    - str or int
        The formatted string with appropriate suffix and currency prefix if specified,
        otherwise returns the original value if it is less than 1000.
    """
    if value >= 1_000_000_000:
        return currency + f'{value / 1_000_000_000:,.0f}B'
    elif value >= 1_000_000:
        return currency + f'{value / 1_000_000:,.0f}M'
    elif value >= 1_000:
        return currency + f'{value / 1_000:,.1f}K'
    else:
        return value