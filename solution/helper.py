import sys
import os
import glob
from pathlib import Path

PROJECT_NAME: str = "Hackathon-2024"

SCRIPT_PATH: Path = Path(os.path.abspath(sys.argv[0])).parent

__PATH= SCRIPT_PATH
while True:
    __PATH = __PATH.parent
    match __PATH.name: 
        case "":
            raise SystemError(f"Parent Folder <{PROJECT_NAME}> not found in parent directory")
        case str(PROJECT_NAME):
            PROJECT_PATH = __PATH
            break
        case _:
            continue

# print(PROJECT_PATH)
SAMPLE_PATH: Path = PROJECT_PATH / "TEST_EXAMPLES"
DATA_PATH: Path = PROJECT_PATH / "data/Rohdaten"

def find_parts(dir: Path, starts_with: str = "mask") -> list[str]:
    pattern: Path = dir / f"**/{starts_with}*"
    return glob.glob(str(pattern), recursive=True)