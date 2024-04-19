from typing import Optional
import requests, zipfile
import pathlib
import argparse
from tqdm import tqdm

def download(output_path: pathlib.Path,
             url: Optional[str] = None):
    """Download a zip data file and unzip the content in the specified directory.

    Keyword arguments:

    output_path: Path where the zip file is going to be unzipped.

    url:         Optional URL of the zip file to be downloaded. If not provided the default URL is used.
    """
        
    if url is None:
        url = "https://urjc-my.sharepoint.com/:u:/g/personal/cristian_romero_urjc_es/EdfbqYpRTvpAgMbdcIklRm0BROzwkEQHobTnZlt3CjyXGw?e=nWu6jw&download=1"

    output_path.mkdir(parents=True, exist_ok=True)

    zip_path = output_path / "temp.zip"
    
    with zip_path.open("wb") as f:
        with requests.get(url, stream=True) as r:

            chunk_size = 4096
            num_chunks = int(r.headers.get("content-length", 0)) // chunk_size
            
            pbar = tqdm(total=num_chunks + 1)
            pbar.set_description("Downloading data")

            for chunk in r.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                pbar.update(1)
                
            pbar.close()
 
    with zipfile.ZipFile(zip_path) as zf:
    
        pbar = tqdm(zf.infolist())
        pbar.set_description("Unzipping data")
        
        for member in pbar:
            try:
                zf.extract(member, output_path)
            except zipfile.error as e:
                pass
                
        pbar.close()
    
    zip_path.unlink()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', type = pathlib.Path, required=True,
                        help='Path where the zip file is going to be unzipped')
                        
    parser.add_argument('--url', type = str,
                        help='Optional URL of the zip file to be downloaded')
    
    args = parser.parse_args()

    download(**vars(args))