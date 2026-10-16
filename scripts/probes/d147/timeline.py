"""timeline.py RUNS_DIR TAG[,TAG...] -- read40k.py's timeline (the contact, the parting, the landings, the tumble, the end
shape error) for the given runs only: read40k.py's main also reads every earlier 40k run it finds by name, and stops at
one whose archive was removed (scene40k_L). The same function, the same JSON lines. CPU."""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("read40k", Path(__file__).with_name("read40k.py"))
read40k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(read40k)
runs = Path(sys.argv[1])
for tag in sys.argv[2].split(","):
    read40k.timeline(runs, tag)
