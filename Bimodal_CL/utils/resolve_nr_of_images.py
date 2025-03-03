import tarfile
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def count_pth_in_single_tar(tar_path):
    """Counts .pth files inside a single tar file."""
    if not os.path.exists(tar_path):
        print(f"File not found: {tar_path}")
        return 0

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            pth_count = sum(1 for member in tar.getmembers() if member.name.endswith(".pth"))
            return tar_path, pth_count
    except Exception as e:
        print(f"Error reading {tar_path}: {e}")
        return 0


def count_pth_files_in_tar_parallel(tar_files, num_workers=8):
    """Parallelized counting of .pth files inside multiple tar files."""
    total_pth_files = 0
    futures = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(tar_files), desc="Processing TAR files", unit="file") as pbar:
            for tar_path in tar_files:
                futures.append(executor.submit(count_pth_in_single_tar, tar_path))

            for future in as_completed(futures):
                result = future.result()
                path = result[0]
                count = result[1]
                print(f"==> COUNT FOR TAR {path}: {count}")

                total_pth_files += count
                pbar.update(1)  # Update progress bar after each completed task

    return total_pth_files


def get_shard_list(start_index=0, end_index=110, base_path="/BS/databases23/CC3M_tar/training/"):
    """Generates a list of tar file paths in the given range."""
    return [
        os.path.join(base_path, f"{i}.tar")
        for i in range(start_index, end_index + 1)
        if os.path.exists(os.path.join(base_path, f"{i}.tar"))
    ]


# Run the parallelized function
tar_files = get_shard_list()
total_count = count_pth_files_in_tar_parallel(tar_files, num_workers=8)
print(f"Total .pth files found: {total_count}")