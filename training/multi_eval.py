import multiprocessing
import time
import queue


# creates marker file, text, timestamp

# process -> attempt create status file -> if exists, check timestamp, if COMPLETE or timestamp < 5 hours skip go next
# finish, mark as complete


def process_file(lock, file_queue):
    while True:
        try:
            lock.acquire()
            try:
                # Attempt to fetch a file from the queue
                file_path = file_queue.get_nowait()
            finally:
                lock.release()

            if file_path is None:
                break

            print(f"Processing {file_path}...")
            # Simulate file processing time
            time.sleep(30)
            print(f"Finished processing {file_path}")

        except queue.Empty:
            # No more files to process
            break


if __name__ == "__main__":
    # List of files to be processed
    files_to_process = ['file1.txt', 'file2.txt', 'file3.txt']
    file_queue = multiprocessing.Queue()

    # Add files to the queue
    for file_path in files_to_process:
        file_queue.put(file_path)

    lock = multiprocessing.Lock()

    # Number of processes
    num_processes = 3

    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=process_file, args=(lock, file_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()