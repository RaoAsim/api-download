import logging
from fastapi import FastAPI, BackgroundTasks
from pathlib import Path
from huggingface_hub import list_repo_refs
from transformers import AutoModelForCausalLM
from bitarray import bitarray
import sqlite3
import json
import asyncio 
from datetime import datetime, timedelta
# SQLite database path
DB_FILE = "model_cache.db"

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

# Initialize the database
def init_db():
    """Set up the SQLite database and table if it does not exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            epoch INTEGER PRIMARY KEY,
            group_data TEXT
        )
    """)
    conn.commit()
    conn.close()

# Cache functions using SQLite
def cache_group(epoch, group):
    """Store group data for a specified epoch in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Store group data as JSON string in the database
    cursor.execute("INSERT OR REPLACE INTO cache (epoch, group_data) VALUES (?, ?)",
                   (epoch, json.dumps(group)))
    conn.commit()
    conn.close()
    logging.info(f"Group data cached for epoch {epoch}")

def get_cached_group(epoch):
    """Retrieve cached group data for a specified epoch from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT group_data FROM cache WHERE epoch = ?", (epoch,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return json.loads(result[0])  # Convert JSON back to list
    return None

# Initialize the database
init_db()

# Parameters
dataset_indices = bitarray(519_000_000)
training_examples_per_miner = 1500
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
        delete_previous_epochs(global_epoch)
    # Generate group object
    # search_start = random.choice(
    #     range(len(dataset_indices) - training_examples_per_miner + 1)
    # )
    # start = dataset_indices.index(
    #     bitarray("0" * training_examples_per_miner), search_start
    # )
    # group = [
    #     i
    #     for i in range(
    #         start, start + training_examples_per_miner
    #     )
    # ]

    # # Cache model and group by global_epoch
    # cache_group(global_epoch, group)  # Save to SQLite
    logging.info(f"Group cached for epoch {global_epoch}")

    return True

# Periodic check for global_epoch update
async def delete_previous_epochs(epoch: int):
    """Deletes refs and cache blobs associated with epochs lower than the given epoch, including .incomplete blobs."""
    deleted_any = False
    for ref in REFS_DIR.iterdir():
        try:
            ref_epoch = int(ref.name)
            if ref_epoch < epoch:
                deleted_any = True
                ref_creation_time = datetime.fromtimestamp(ref.stat().st_ctime)
                for blob in CACHE_DIR.iterdir():
                    blob_creation_time = datetime.fromtimestamp(blob.stat().st_ctime)
                    if ref_creation_time <= blob_creation_time <= ref_creation_time + timedelta(minutes=10):
                      blob.unlink()
                
                with open(ref, "r") as file:
                    ref_value = file.read().strip() 
                    snapshot_path = SNAPSHOTS_DIR / ref_value
                    if snapshot_path.exists() and snapshot_path.is_dir():
                        # Recursively delete the snapshot folder
                        for child in snapshot_path.iterdir():
                            child.unlink()  # Delete file inside the snapshot
                        snapshot_path.rmdir()  # Remove the snapshot folder
                        logging.info(f"Deleted snapshot folder {snapshot_path} for incomplete blob")

                ref.unlink()
                logging.info(f"Deleted ref for epoch {ref_epoch}")

        except ValueError:
            continue  # Ignore non-integer filenames in the refs directory

    if not deleted_any:
        logging.info(f"No refs with smaller epochs than {epoch} were found")

    # Delete blobs with .incomplete in their name
    for blob in CACHE_DIR.iterdir():
        if ".incomplete" in blob.name:
            try:
                blob.unlink()
                logging.info(f"Deleted incomplete blob {blob.name}")
            except Exception as e:
                logging.error(f"Error deleting incomplete blob {blob.name}: {e}")
    
    logging.info(f"Deletion Completed")
            

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

@app.get("/get_group/{epoch}")
async def get_group(epoch: int):
    """Endpoint to retrieve cached group by epoch from SQLite database."""
    cached_data = get_cached_group(epoch)
    if cached_data:
        return {"group": cached_data}
    else:
        return {"error": "Data not found for the specified epoch"}
