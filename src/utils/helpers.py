import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from vector_database.exceptions import APIKeyError


def get_api_key(env_path: str = "/secrets.env", key_name: str = "PINECONE_API_KEY") -> str:
    """
    Retrieves the API key from a specified .env file.

    This function loads environment variables from the given .env file and retrieves the value
    associated with the specified key name. It raises custom exceptions if the file is not found
    or if the key is missing.

    Args:
        env_path (str): The path to the .env file. Defaults to ".env".
        key_name (str): The name of the API key variable to retrieve. Defaults to "API_KEY".

    Returns:
        str: The API key retrieved from the .env file.

    Raises:
        FileNotFoundError: If the .env file does not exist at the specified path.
        APIKeyError: If the API key is not found in the .env file.
    """

    # Define the path to the .env file using pathlib for cross-platform compatibility
    env_file = Path(os.getcwd() + env_path)

    # Check if the .env file exists
    if not env_file.is_file():
        raise FileNotFoundError(f"The .env file was not found at path: {env_path}")

    # Load the environment variables from the .env file
    load_dotenv(dotenv_path=env_file)

    # Retrieve the API key from the environment variables
    api_key: Optional[str] = os.getenv(key_name)

    # Validate that the API key exists
    if not api_key:
        raise APIKeyError(f"The key '{key_name}' was not found in the .env file.")

    return api_key
