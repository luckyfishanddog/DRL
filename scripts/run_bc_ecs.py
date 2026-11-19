"""Run fixed BC-ECS experiments or summarize their saved results."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_saeax import configure_cpu
configure_cpu()

from mrta_search.bc_ecs import main


if __name__ == '__main__':
    main()
