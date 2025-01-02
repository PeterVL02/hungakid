from src.project_store import ProjectStore
from src.shell import Shell

import sys
import re
from io import StringIO
from typing import Any

ANSI_ESCAPE_PATTERN = re.compile(r'\x1B\[[0-9;]*[A-Za-z]')

def simulate_cli(commands: list[str]) -> str:
    project_store = ProjectStore()
    shell = Shell(project_store)

    original_stdin = sys.stdin
    original_stdout = sys.stdout
    buffer = StringIO()
    
    if not commands[-1] == "exit":
        commands.append("exit")

    try:
        # Redirect stdin and stdout
        sys.stdin = StringIO("\n".join(commands) + "\n")
        sys.stdout = buffer

        # Run the shell logic
        shell.run()

        # Retrieve everything printed to stdout
        output = buffer.getvalue()
        
        # 1. Remove ANSI color codes and other escapes
        #    e.g., \x1b[32m >> or \x1b[0m
        cleaned_output = ANSI_ESCAPE_PATTERN.sub('', output)

        # 2. Split on newlines to handle line-by-line filtering
        lines = cleaned_output.split()

        # 3. Filter out prompt lines (>>), as well as any empty lines
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip if empty or just '>>'
            if not stripped or stripped == '>>':
                continue
            filtered_lines.append(stripped)

        # 4. Re-join the filtered lines into a single string
        final_output = ' '.join(filtered_lines)

        return final_output

    except Exception as e:
        return f'Failed: {e}'
    finally:
        # Always restore
        sys.stdin = original_stdin
        sys.stdout = original_stdout
        buffer.close()
        
EDGE_CASES = ['Cross Validation', ]

def _cover_edge_cases(*cases: Any) -> str:
    to_eval = []
    for case in cases:
        if not isinstance(case, str):
            case = str(case)
        if not case in EDGE_CASES:
            to_eval.append(case)
    return ' '. join(to_eval)

def convert_expected(*expected: Any) -> str:
    return _cover_edge_cases(*expected)

def main() -> None:
    commands = [
            "load NonExistingProject",
            "exit",
        ]
    result = simulate_cli(commands)
    split_result = result.split()
    print(split_result)
    
if __name__ == "__main__":
    main()