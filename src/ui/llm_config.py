class LLMConfig:
    """
    A class to store and manage the configuration parameters for the Language Model (LM).
    It provides the ability to update parameters such as the generation temperature and
    the maximum number of tokens to generate.
    """

    def __init__(self, temperature: float = 0.7, max_tokens: int = 500) -> None:
        """
        Initialize the LLMConfig with default or specified parameters.

        Args:
            temperature (float): The creativity parameter for the LM's responses.
                                 Higher values result in more creative and varied outputs. Default is 0.7.
            max_tokens (int): The maximum number of tokens to generate in the LM's response. Default is 500.
        """
        self.temperature = temperature
        self.max_tokens = max_tokens

    def update_temperature(self, new_temp: float) -> None:
        """
        Update the LM's response temperature.

        Args:
            new_temp (float): The new temperature value.
        """
        self.temperature = new_temp

    def update_max_tokens(self, new_max_tokens: int) -> None:
        """
        Update the LM's maximum number of tokens to generate.

        Args:
            new_max_tokens (int): The new maximum tokens value.
        """
        self.max_tokens = new_max_tokens
