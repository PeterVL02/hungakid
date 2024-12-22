from dataclasses import dataclass

from src.commands.command_factory import cmd_exists, execute_cmd
from src.commands.project_store_protocol import Model


@dataclass
class Command:
    cmd: str
    args: list[str | int | list[str]]
    kwargs: dict[str, str | int | list[str]]

    @staticmethod
    def from_string(command: str) -> "Command":
        parts = command.split()
        try:
            cmd, raw_args = parts[0], parts[1:]
        except IndexError:
            raise ValueError("Command must be first argument. Empty commands are not allowed.")
        if not cmd_exists(cmd = cmd):
            raise ValueError(f"Command {cmd} does not exist.")

        parsed_args: list[str | int | list[str]] = []
        parsed_kwargs: dict[str, str | int | list[str]] = {}
        for arg in raw_args:
            try:
                if '=' in arg:
                    key, value = arg.split('=')
                    if value.isnumeric():
                        parsed_kwargs[key] = int(value)
                    else:
                        parsed_kwargs[key] = value
                elif arg.isnumeric():
                    parsed_args.append(int(arg))
                else:
                    parsed_args.append(arg)
            except ValueError:
                pass

        return Command(cmd, parsed_args, parsed_kwargs)

    def execute(self, model: Model) -> None:
        execute_cmd(self.cmd, model, *self.args, **self.kwargs)
        print()