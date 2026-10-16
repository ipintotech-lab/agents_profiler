"""Rough list prices for the dollar scenarios, $ per 1M tokens
(input, output).

These are estimates for scenario math only -- providers change prices
and the traces don't say which model ran, so the report always labels
the result as an estimate and the caller can always override with exact
prices (``price_input_per_mtok`` / ``price_output_per_mtok``).

Looking up an unknown model fails loudly (listing what is known) rather
than guessing a price.
"""

# label shown to the customer -> (input $/Mtok, output $/Mtok)
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-3.5": (0.80, 4.0),
    "gpt-5": (1.25, 10.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
}


def lookup_price(model: str) -> tuple[float, float]:
    """(input $/Mtok, output $/Mtok) for a model name, case-insensitive.

    Raises ValueError for unknown names -- never guesses a price.
    """
    key = model.strip().lower()
    for name, prices in PRICE_TABLE.items():
        if key == name.lower():
            return prices
    raise ValueError(
        f"no price known for model '{model}' -- known models: {', '.join(sorted(PRICE_TABLE))} "
        "(or pass price_input_per_mtok / price_output_per_mtok directly)"
    )


# Example rates for the "what would this save in dollars" line, shown when
# the traces record no cost: three price classes instead of one guess.
SCENARIOS: list[tuple[str, float, float]] = [
    ("Haiku-class", 0.80, 4.00),
    ("Sonnet-class", 3.00, 15.00),
    ("Opus-class", 15.00, 75.00),
]
