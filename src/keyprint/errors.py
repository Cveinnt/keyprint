"""Public input errors with safe, actionable details."""


class InputLimitError(ValueError):
    """Tokenized input exceeds a backend limit, before sampling starts.

    ``max_tokens`` is the reserved response budget for a context limit. When
    omitted, ``limit`` applies to input tokens alone. Counts include the model's
    chat template and, for rewriting, the instructions around the source text.
    """

    def __init__(self, *, input_tokens: int, limit: int, max_tokens: int | None = None):
        if any(type(n) is not int or n < 1 for n in (input_tokens, limit)):
            raise ValueError("input_tokens and limit must be positive integers")
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer when supplied")
        if input_tokens + (max_tokens or 0) <= limit:
            raise ValueError("token counts must exceed the declared limit")
        self.input_tokens, self.limit, self.max_tokens = input_tokens, limit, max_tokens
        if max_tokens is None:
            message = (f"Input has {input_tokens} tokens; the supported input limit is {limit}. "
                       "Shorten the input.")
        else:
            advice = ("Shorten the input." if input_tokens >= limit else
                      "Shorten the input or lower max_tokens.")
            message = (f"Input has {input_tokens} tokens and the response budget is {max_tokens}; "
                       f"the context limit is {limit}. {advice}")
        super().__init__(message)
