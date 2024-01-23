import os
from huggingface_hub import hf_hub_download
import urllib.request

def download_model_from_hf(repo_id, filename, local_dir, cache_dir):
    hf_hub_download(repo_id=repo_id, filename=filename,local_dir=local_dir, cache_dir=cache_dir)

def download_chinook_db():
    #TODO: need to correct this code to download the database
    os.makedirs(r".\db", exist_ok=True)
    urllib.request.urlretrieve("https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip", r".\db\chinook.zip")

if __name__ == "__main__":

    #Download Model
    repo_id="TheBloke/Llama-2-7B-GGUF"
    filename="llama-2-7b.Q4_K_M.gguf"
    local_dir = r".\models"
    cache_dir =  r".\models\cache"
    download_model_from_hf(repo_id, filename, local_dir, cache_dir)

    # download_chinook_db()
