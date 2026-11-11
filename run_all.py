"""Run each available recipe in its own process and output directory."""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
# Aggiungeremo qui gli altri due script quando saranno disponibili.
EXPERIMENTS = ["experiments/01_hybrid_symmetric.py"]


def main():
    parser = argparse.ArgumentParser(description="Esegue tutte le ricette disponibili in sequenza.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--list", action="store_true", help="Elenca gli script senza eseguirli.")
    args = parser.parse_args()
    for relative_path in EXPERIMENTS:
        if not (ROOT / relative_path).is_file():
            parser.error(f"Script mancante: {relative_path}")
    if args.list:
        print("\n".join(EXPERIMENTS))
        return
    if args.data_dir is None:
        parser.error("Indica la cartella del dataset con --data-dir PATH.")
    output_directory = args.output_dir.resolve()
    for index, relative_path in enumerate(EXPERIMENTS, 1):
        script = ROOT / relative_path
        command = [sys.executable, "-u", str(script), "--output-dir",
                   str(output_directory / script.stem)]
        command += ["--data-dir", str(args.data_dir.resolve())]
        for flag in ("cv", "save_models"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        print(f"[{index}/{len(EXPERIMENTS)}] {script.name}", flush=True)
        subprocess.run(command, check=True)
    print(f"Completato. Risultati: {output_directory}", flush=True)


if __name__ == "__main__":
    main()
