class CLIException(Exception):
    """
    Custom exception class for command line interface errors.

    Args:
        message (str): The error message.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"CliException: {self.message}"