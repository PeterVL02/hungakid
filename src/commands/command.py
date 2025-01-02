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
        # 1) Split using shlex
        if not command.islower():
            print("Warning: Command will be converted to lowercase.")
        command = command.lower()
        parts = shlex.split(command)
        if not parts:
            raise ValueError("Command must not be empty.")

        cmd, raw_args = parts[0], parts[1:]

        if not cmd_exists(cmd):
            raise ValueError(f"Command '{cmd}' does not exist.")

        # 2) Preprocess raw_args to recombine bracketed items
        #    Example: "[apples," and "bananas]" -> "[apples, bananas]"
        processed_args = Command._recombine_bracketed_tokens(raw_args)

        parsed_args: list[str | int | list[str]] = []
        parsed_kwargs: dict[str, str | int | list[str]] = {}

        # 3) Normal parsing with the bracketed tokens rejoined
        iterator = iter(processed_args)
        for token in iterator:
            if token.startswith("--"):
                # long-form option (--key val or --key=val)
                key = token[2:]
                if "=" in key:
                    key, value = key.split("=", 1)
                else:
                    value = next(iterator, None)
                    if value is None or value.startswith('-'):
                        raise ValueError(f"Option '{token}' requires a value.")
                parsed_kwargs[key] = Command._parse_value(value)

            elif token.startswith("-"):
                # short-form option (-k val)
                key = token[1:]
                value = next(iterator, None)
                if value is None or value.startswith('-'):
                    raise ValueError(f"Option '{token}' requires a value.")
                parsed_kwargs[key] = Command._parse_value(value)

            else:
                # positional argument
                parsed_args.append(Command._parse_value(token))

        return Command(cmd, parsed_args, parsed_kwargs)

    @staticmethod
    def _recombine_bracketed_tokens(tokens: list[str]) -> list[str]:
        """
        Recombine any tokens that start with '[' and don't end with ']' until
        we reach a token that does, effectively gluing them back together.
        """
        result = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            # If a token starts with '[' but does not end with ']', 
            # accumulate tokens until we find the one that ends with ']'.
            if token.startswith("[") and not token.endswith("]"):
                bracketed_parts = [token]
                i += 1
                # Keep appending tokens until we find one that ends with ']'
                while i < len(tokens) and not tokens[i].endswith("]"):
                    bracketed_parts.append(tokens[i])
                    i += 1
                # If we haven't run out of tokens, include the closing part
                if i < len(tokens):
                    bracketed_parts.append(tokens[i])
                    i += 1
                # Join them with space (or no space, depending on your needs)
                combined = " ".join(bracketed_parts)
                result.append(combined)
            else:
                result.append(token)
                i += 1
        return result

    @staticmethod
    def _parse_value(value: str) -> str | int | list[str]:
        """Parse individual value into int, list, or string."""
        # 1) Numeric check (include negative numbers)
        if value.lstrip('-').isnumeric():
            return int(value)

        # 2) Try literal_eval (in case the value is like ["apples", "bananas"])
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        # 3) Manual check for bracketed list of unquoted items:
        #    e.g. [apples, bananas]
        if value.startswith("[") and value.endswith("]"):
            content = value[1:-1].strip()
            # If this is a bracketed expression that has no quotes inside,
            # parse it manually
            # e.g. if content == "apples, bananas"
            if content and not any(c in content for c in ("'", '"')):
                return [item.strip() for item in content.split(",")]

        # 4) Otherwise, just return the string
        return value


    def execute(self, model: Model) -> None:
        execute_cmd(self.cmd, model, *self.args, **self.kwargs)
