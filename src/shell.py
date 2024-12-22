from src.commands.command import Command
from src.commands.project_store_protocol import Model
from colorama import Fore, Back, Style


class Shell:
    def __init__(self, model: Model) -> None:
        self.model = model

    def run(self) -> None:
        while True:
            user_input = input(Fore.GREEN + ">> " + Style.RESET_ALL)
            if not user_input: continue
            try:
                for subcommand in user_input.split(';'):
                    if subcommand == "exit":
                        return
                    command = Command.from_string(subcommand)
                    command.execute(self.model)
            except (TypeError, ValueError, IndexError) as e:
                self.display_message("Incorrect args provided for command")
                self.display_log(e.with_traceback(None))

    def display_message(self, message: str) -> None:
        print(Fore.RED + message)
        print(Style.RESET_ALL)

    def display_log(self, exception: Exception) -> None:
        print(exception)