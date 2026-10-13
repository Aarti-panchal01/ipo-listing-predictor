"""Print the text outputs of an executed notebook (for reviewing results)."""
import sys

import nbformat

sys.stdout.reconfigure(encoding="utf-8")
nb = nbformat.read(sys.argv[1], as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            print(f"--- [{i}] stream\n{o['text'][:3000]}")
        elif "data" in o and "text/plain" in o["data"]:
            print(f"--- [{i}] result\n{o['data']['text/plain'][:3000]}")
        elif o.get("output_type") == "error":
            print(f"--- [{i}] ERROR {o['ename']}: {o['evalue']}")
