import logging
from fastapi import FastAPI, BackgroundTasks
from pathlib import Path
from huggingface_hub import list_repo_refs
from transformers import AutoModelForCausalLM
from bitarray import bitarray
import json
import asyncio 
from datetime import datetime, timedelta
# SQLite database path


# Initialize FastAPI app
app = FastAPI()
REFS_DIR = Path("/root/.cache/huggingface/hub/models--distributed--optimized-gpt2-2b/refs").expanduser()
CACHE_DIR = Path("/root/.cache/huggingface/hub/models--distributed--optimized-gpt2-2b/blobs").expanduser()
SNAPSHOTS_DIR = Path("/root/.cache/huggingface/hub/models--distributed--optimized-gpt2-2b/snapshots").expanduser()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # Log to console, captured by PM2
    ]
)



local_epoch = None

# Get global epoch from Hugging Face repo
async def get_global_epoch():
    """Fetch the highest epoch tag from the Hugging Face repository asynchronously."""
    refs = await asyncio.to_thread(list_repo_refs, "distributed/optimized-gpt2-2b", repo_type="model")
    global_epoch = max([int(tag.name) for tag in refs.tags]) if refs.tags else None
    logging.info(f"Global epoch retrieved: {global_epoch}")
    return global_epoch

# Download model and update group in cache
async def model_download(global_epoch):
    """Download the model, generate group, and cache it."""
    # Download the model
    model = await asyncio.to_thread(
        AutoModelForCausalLM.from_pretrained,
        "distributed/optimized-gpt2-2b",
        revision=str(global_epoch),  # This must be a valid git revision
        trust_remote_code=True
    )
    logging.info(f"Model downloaded for epoch {global_epoch}")
    if global_epoch > 0:
      await  delete_previous_epochs(global_epoch)
    logging.info(f"Group cached for epoch {global_epoch}")

    return True


import shutil
import os
import time

async def delete_previous_epochs(epoch: int):
    """Deletes refs and cache blobs associated with epochs lower than the given epoch, including .incomplete blobs."""
    deleted_any = False

    for ref in REFS_DIR.iterdir():
        try:
            ref_epoch = int(ref.name)
            if ref_epoch < epoch:
                deleted_any = True
                ref_creation_time = datetime.fromtimestamp(ref.stat().st_ctime)

                # Delete related blobs (Retry if file is in use)
                for blob in CACHE_DIR.iterdir():
                    blob_creation_time = datetime.fromtimestamp(blob.stat().st_ctime)
                    if ref_creation_time <= blob_creation_time <= ref_creation_time + timedelta(minutes=10):
                        await safe_delete(blob)

                # Read ref value and delete associated snapshot
                with open(ref, "r") as file:
                    ref_value = file.read().strip()
                    snapshot_path = SNAPSHOTS_DIR / ref_value
                    if snapshot_path.exists():
                        await safe_delete(snapshot_path, is_dir=True)

                # Finally, delete ref itself
                await safe_delete(ref)

        except ValueError:
            continue  # Ignore non-integer filenames
        except Exception as e:
            logging.error(f"Error processing ref {ref}: {e}")

    if not deleted_any:
        logging.info(f"No refs with smaller epochs than {epoch} were found")

    # Delete blobs with .incomplete in their name (Retry if necessary)
    for blob in CACHE_DIR.iterdir():
        if ".incomplete" in blob.name:
            await safe_delete(blob)

    logging.info("Deletion Completed")

async def safe_delete(path, is_dir=False, retries=3, delay=2):
    """Safely deletes a file or directory with retries to prevent permission errors."""
    for attempt in range(retries):
        try:
            if is_dir:
                shutil.rmtree(path)  # Ensures all subfiles are deleted
                logging.info(f"Deleted directory: {path}")
            else:
                path.unlink()
                logging.info(f"Deleted file: {path}")
            return True
        except FileNotFoundError:
            logging.warning(f"File not found: {path}, skipping.")
            return True
        except PermissionError as e:
            logging.warning(f"Permission denied: {path}, retrying... [{attempt + 1}/{retries}]")
            time.sleep(delay)
        except OSError as e:
            logging.error(f"Error deleting {path}: {e}")
            time.sleep(delay)
    
    logging.error(f"Failed to delete {path} after {retries} attempts")
    return False

            

async def periodic_check():
    """Background task to periodically check for updates."""
    global local_epoch
    while True:
        try:
            global_epoch =await get_global_epoch()
            if global_epoch is not None and global_epoch != local_epoch:
                # Update the model if there's a new global epoch
                await model_download(global_epoch)
                local_epoch = global_epoch
                logging.info(f"Updated local epoch to {local_epoch}")
            else:
                logging.info("No update required.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        
        # Wait for 5 minutes before the next check
        await asyncio.sleep(200)

# Start the periodic check in a background thread
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_check())

@app.post("/delete_cache/{epoch}")
async def delete_cache(epoch: int, background_tasks: BackgroundTasks):
    """API endpoint to trigger deletion of cache and refs for previous epochs, including .incomplete blobs."""
    background_tasks.add_task(delete_previous_epochs, epoch)
    return {"status": "Deletion initiated for previous epochs"}

@app.get("/status")
async def get_status():
    """Simple endpoint that returns true."""
    return {"status": True}

