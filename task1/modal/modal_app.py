# ============================================================
# modal_app.py
#
# This file defines the Modal app.
#
# It:
# 1. Creates a Modal container image
# 2. Installs dependencies
# 3. Mounts a persistent Modal volume
# 4. Runs evaluate.py on a GPU
# ============================================================

import modal

# ============================================================
# CREATE MODAL APP
# ============================================================

app = modal.App("llm-eval-app")


# ============================================================
# CREATE CONTAINER IMAGE
#
# This image is the environment that runs remotely.
# ============================================================

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "pyyaml",
        "tqdm",
        "sentencepiece",
    )
)


# ============================================================
# CREATE PERSISTENT VOLUME
#
# This stores:
# - adapters
# - data
# - outputs
#
# Files persist between runs.
# ============================================================

volume = modal.Volume.from_name(
    "llm-eval-volume",
    create_if_missing=True
)


# ============================================================
# GPU FUNCTION
#
# This function runs remotely on Modal.
# ============================================================

@app.function(
    image=image,

    # GPU TYPE
    gpu="L4",

    # Maximum runtime
    timeout=60 * 60 * 12,

    # Mount volume inside container
    volumes={
        "/root/project": volume
    },
)
def run_evaluation():

    # Import INSIDE remote function
    # so imports happen on Modal machine

    import sys

    # Add mounted project directory to Python path
    sys.path.append("/root/project")

    # Import evaluation function
    from evaluate import main

    # Run evaluation
    main(device="cuda")


# ============================================================
# LOCAL ENTRYPOINT
#
# This runs when you execute:
#
#     modal run modal_app.py
# ============================================================

@app.local_entrypoint()
def main():

    # Trigger remote GPU execution
    run_evaluation.remote()