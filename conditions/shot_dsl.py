import json
def load_shots(path:str):
    if not path: return []
    with open(path) as f:
        obj=json.load(f)
    return obj.get("scenes",[])