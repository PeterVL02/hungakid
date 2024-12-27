from dataclasses import dataclass
import shlex
import ast

from src.commands.command_factory import cmd_exists, execute_cmd
from src.commands.project_store_protocol import Model


@dataclass
class Command:
    cmd: str
    args: list[str | int | list[str]]
    kwargs: dict[str, str | int | list[str]]

    @staticmethod
    def from_string(command: str) -> "Command":
        parts = shlex.split(command) 
        if not parts:
            raise ValueError("Command must not be empty.")

        cmd, raw_args = parts[0], parts[1:]

        if not cmd_exists(cmd):
            raise ValueError(f"Command '{cmd}' does not exist.")

        parsed_args: list[str | int | list[str]] = []
        parsed_kwargs: dict[str, str | int | list[str]] = {}

        iterator = iter(raw_args)
        for token in iterator:
            if token.startswith("--"):
                key = token[2:]
                if '=' in key:
                    key, value = key.split('=', 1)
                else:
                    value = next(iterator, None)
                    if value is None or value.startswith('-'):
                        raise ValueError(f"Option '{token}' requires a value.")
                parsed_kwargs[key] = Command._parse_value(value)
            elif token.startswith("-"):  
                key = token[1:]
                value = next(iterator, None)
                if value is None or value.startswith('-'):
                    raise ValueError(f"Option '{token}' requires a value.")
                parsed_kwargs[key] = Command._parse_value(value)
            else:
                parsed_args.append(Command._parse_value(token))

        return Command(cmd, parsed_args, parsed_kwargs)

    @staticmethod
    def _parse_value(value: str) -> str | int | list[str]:
        """Parse individual value into int, list, or string."""
        if value.isnumeric():
            return int(value)
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        return value

    def execute(self, model: Model) -> None:
        execute_cmd(self.cmd, model, *self.args, **self.kwargs)
