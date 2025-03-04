import os
import time


def get_available_file(processed_pkl_dir, status_dir, pred_dir, max_wait=7200):
    for filename in os.listdir(processed_pkl_dir):
        if not filename.endswith('.pkl'):
            continue

        filepath = os.path.join(processed_pkl_dir, filename)
        txt_filename = filename.replace('.pkl', '.txt')
        txt_filepath = os.path.join(status_dir, txt_filename)

        pred_filename = filename.replace('.pkl', '.csv')
        pred_filepath = os.path.join(pred_dir, pred_filename)

        if os.path.exists(pred_filepath):
            continue

        now = int(time.time())
        available = True
        if os.path.exists(txt_filepath):
            with open(txt_filepath, 'r') as file:
                file_contents = file.readline().strip()
                try:
                    file_time = int(file_contents)
                    sec_delta = now - file_time
                    if sec_delta <= max_wait:
                        available = False
                except ValueError:
                    available = False

        if not available:
            continue

        with open(txt_filepath, 'w') as file:
            file.write(str(now))

        return filename, filepath





