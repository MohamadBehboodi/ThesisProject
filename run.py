from pathlib import Path
import sys
from thesis.cli import main
if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv += ['preflight', '--config', str(Path(__file__).parent / 'configs/default.json')]
    main()
