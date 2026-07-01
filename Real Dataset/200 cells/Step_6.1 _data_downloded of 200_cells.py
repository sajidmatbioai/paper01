import requests
import urllib.request
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

NWB_DIR = '/kaggle/working/nwb_cells/'
os.makedirs(NWB_DIR, exist_ok=True)

# 200 cells fetch karo Allen Brain API se
url = (
    "http://api.brain-map.org/api/v2/data/query.json?"
    "criteria=model::Specimen,"
    "rma::criteria,[is_cell_specimen$eqtrue],"
    "rma::include,ephys_result,"
    "rma::options[num_rows$eq200][order$eqid]"
)

response = requests.get(url, timeout=60)
data     = response.json()
cells    = [c for c in data['msg']
            if isinstance(c, dict) and c.get('ephys_result') is not None]
cell_ids = [c['id'] for c in cells]
print(f"Cells found: {len(cell_ids)}")

def download_cell(cell_id):
    fname = f'{NWB_DIR}cell_{cell_id}.nwb'
    if os.path.exists(fname):
        return f"{cell_id} exists"
    try:
        api_url = (
            f"http://api.brain-map.org/api/v2/data/query.json?"
            f"criteria=model::Specimen,"
            f"rma::criteria,[id$eq{cell_id}],"
            f"rma::include,ephys_result("
            f"well_known_files(well_known_file_type[name$eqNWBDownload]))"
        )
        r    = requests.get(api_url, timeout=30)
        d    = r.json()
        dl   = ("http://api.brain-map.org" +
                d['msg'][0]['ephys_result']['well_known_files'][0]['download_link'])
        urllib.request.urlretrieve(dl, fname)
        size = os.path.getsize(fname)/1024/1024
        return f"{cell_id} done — {size:.1f} MB"
    except Exception as e:
        return f"{cell_id} FAILED: {e}"

completed = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(download_cell, cid): cid for cid in cell_ids}
    for future in as_completed(futures):
        result = future.result()
        completed += 1
        if completed % 20 == 0 or 'FAILED' in result:
            print(f"  [{completed}/{len(cell_ids)}] {result}")

total_size = sum(
    os.path.getsize(f'{NWB_DIR}{f}')
    for f in os.listdir(NWB_DIR) if f.endswith('.nwb')
)/1024/1024/1024

print(f"\nTotal: {total_size:.2f} GB")
print(f"Files: {len(os.listdir(NWB_DIR))}")