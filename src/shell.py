from src.commands.command import Command
from src.commands.project_store_protocol import Model
from colorama import Fore, Style

class Shell:
    def __init__(self, model: Model) -> None:
        self.model = model

    def process_cmd(self, cmd: str) -> bool:
        """Processes command. If command is "exit", returns False.

        Args:
            cmd (str): Command to process.

        Returns:
            bool: True if command is not "exit", False otherwise.
        """
        if cmd == "exit":
            return False
        try:
            command = Command.from_string(cmd)
            command.execute(self.model)
        except (TypeError, ValueError) as e:
            raise e
        return True


    def run(self) -> None:
        """Runs the shell."""
        while True:
            user_input = input(Fore.GREEN + ">> " + Style.RESET_ALL)
            if not user_input: continue
            try:
                for subcommand in user_input.split(';'):
                    keep_alive = self.process_cmd(subcommand)
                    if not keep_alive:
                        return
            except (TypeError, ValueError) as e:
                self.display_message("Error:")
                self.display_log(e.with_traceback(None))

    def display_message(self, message: str) -> None:
        """
        Displays a message in red color.
        Args:
            message (str): The message to be displayed.
        """
        print(Fore.RED + message + Style.RESET_ALL)

    def display_log(self, exception: Exception) -> None:
        """
        Displays the provided exception message in red color.
        Args:
            exception (Exception): The exception whose message is to be displayed.
        """
        print(Fore.RED + str(exception) + Style.RESET_ALL)