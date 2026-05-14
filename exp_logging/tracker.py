from pathlib import Path
import json

class ScalarTracker:
    def __init__(self, work_dir:Path):
        self.path = work_dir / "metrics" / "scalars.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_scalar(self, tag:str, value:float, step:int):
        self.path.write_text(self.path.read_text() + json.dumps({"tag":tag,"value":float(value),"step":int(step)})+"\n" if self.path.exists()
                             else json.dumps({"tag":tag,"value":float(value),"step":int(step)})+"\n")
