from src.commands.command import Command
from src.commands.project_store_protocol import Model
from src.cliresult import CLIResult

from colorama import Fore, Style

class Shell:
    def __init__(self, model: Model) -> None:
        self.model = model

    def process_cmd(self, cmd: str) -> tuple[bool, CLIResult | None]:
        """Processes command. If command is "exit", returns False.

        Args:
            cmd (str): Command to process.

        Returns:
            bool: True if command is not "exit", False otherwise.
        """
        if cmd == "exit":
            return False, None
        try:
            command = Command.from_string(cmd)
            result = command.execute(self.model)
        except (TypeError, ValueError, AssertionError) as e:
            raise e
        return True, result


    def run(self) -> None:
        """Runs the shell."""
        while True:
            user_input = input(Fore.GREEN + ">> " + Style.RESET_ALL)
            if not user_input: continue
            try:
                for subcommand in user_input.split(';'):
                    keep_alive, result = self.process_cmd(subcommand)
                    if not keep_alive:
                        return
                    if result is None: continue
                    if result.warning:
                        self.display_message(result.warning.strip(), c = result.c_warn)
                    if result.result:
                        self.display_message(result.result, c = result.c_message)
                    if result.note:
                        self.display_message(result.note.strip(), c = result.c_note)
            except (ValueError, AssertionError, TypeError, AttributeError) as e:
                self.display_message("Error:")
                self.display_log(e)
            

    def display_message(self, message: str, c: str = Fore.RED) -> None:
        """
        Displays a message in red color.
        Args:
            message (str): The message to be displayed.
        """
        print(c + message + Style.RESET_ALL)

    def display_log(self, exception: Exception, c: str = Fore.RED) -> None:
        """
        Displays the provided exception message in red color.
        Args:
            exception (Exception): The exception whose message is to be displayed.
        """
        print(c + str(exception) + Style.RESET_ALL)