from pydantic import BaseModel
import yaml

class Config(BaseModel):
    model_path: str
    video_source: int|str
    vehicle_classes: list[str]
    confidence_threshold: float

def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        raw_data = yaml.safe_load(f)
    return Config(**raw_data)