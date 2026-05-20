import modal
from pathlib import Path

volume = modal.Volume.from_name("llm-eval-volume")

output_dir = Path("downloaded_outputs")
output_dir.mkdir(exist_ok=True)

volume.reload()

for file in volume.listdir("/outputs"):
    local_path = output_dir / file.path.split("/")[-1]

    with open(local_path, "wb") as f:
        f.write(volume.read_file(file.path))

print("Downloaded outputs!")