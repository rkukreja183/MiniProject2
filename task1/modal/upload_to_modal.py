# ============================================================
# upload_to_modal.py
#
# Uploads:
# - evaluate.py
# - data/
# - finetuned_models_new/
#
# into the Modal persistent volume.
#
# Run locally:
#
#     python upload_to_modal.py
# ============================================================

import modal

# ============================================================
# CONNECT TO MODAL VOLUME
# ============================================================

volume = modal.Volume.from_name(
    "llm-eval-volume",
    create_if_missing=True
)

# ============================================================
# UPLOAD FILES
# ============================================================

with volume.batch_upload() as batch:

    # Upload evaluation script
    batch.put_file(
        "evaluate.py",
        "/evaluate.py"
    )

    # Upload evaluation questions
    batch.put_directory(
        "data",
        "/data"
    )

    # Upload all finetuned adapters
    batch.put_directory(
        "finetuned_models_new",
        "/finetuned_models_new"
    )

print("Upload complete!")