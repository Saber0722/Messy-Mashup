"""
Project entry point.

Usage:
    python main.py build       # extract mels + create splits
    python main.py train       # train MultiBranchCRNN
    python main.py evaluate    # evaluate best checkpoint on test set
    python main.py infer       # generate submission.csv
"""

import sys
from pathlib import Path
from rich.console import Console

console = Console()

COMMANDS = {
    "build":    "scripts/build_dataset.py",
    "train":    "scripts/train_multibranch.py",
    "evaluate": "scripts/evaluate.py",
    "infer":    "scripts/infer_test.py",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        console.print("[bold]Usage:[/bold] python main.py [build|train|evaluate|infer]")
        console.print("\nAvailable commands:")
        for cmd, script in COMMANDS.items():
            console.print(f"  [cyan]{cmd:<12}[/cyan] → {script}")
        sys.exit(1)

    cmd = sys.argv[1]
    script = Path(__file__).parent / COMMANDS[cmd]

    assert script.exists(), f"Script not found: {script}"

    console.rule(f"[bold cyan]{cmd.upper()}")

    # Execute the script in-process by running its main()
    import importlib.util, types

    spec = importlib.util.spec_from_file_location("__script__", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["__script__"] = mod
    spec.loader.exec_module(mod)   # type: ignore[union-attr]

    if hasattr(mod, "main"):
        mod.main()


if __name__ == "__main__":
    main()