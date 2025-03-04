#! /usr/bin/env python3

"""
check_download_completed.py
    Looks in a fits folder for a sector and determines if there are missing
    files that should have been downloaded but weren't. If they weren't
    downloaded then it will download them.
"""

import argparse
import os
import subprocess
from astropy.io import fits

def main(fits_dir, sectors):
    """
    main
        Drives the script
    """
    for sector in sectors:
        full_dir = os.path.join(fits_dir, sector)
        downloaded_fits, curl_file = get_files(full_dir)
        curl_commands = get_curl_commands(full_dir, curl_file)
        missing_fits_commands, bad_files, bad_file_sizes = find_missing_bad_fits(full_dir, downloaded_fits, curl_commands)
        
        print("Downloading {} files for {}".format(len(missing_fits_commands), sector))
        
        chunk_size = 128
        for i in range(0, len(missing_fits_commands), chunk_size):
            command_chunk = missing_fits_commands[i:i+chunk_size]
            command_string = "\n".join(command_chunk)
            xargs_command = f"echo '{command_string}' | xargs -P 4 -I % sh -c '%'"
            subprocess.run(xargs_command, shell=True, cwd=full_dir)

def delete_bad_files(fits_dir, bad_files):
    print("Removing {} bad files".format(len(bad_files)))
    for bad_file in bad_files:
        if bad_file is None:
            continue

        file_path = os.path.join(fits_dir, bad_file)
        if os.path.exists(file_path):
            os.remove(file_path)

def download_missing_fits(fits_dir, missing_fits_commands):
    """
    Runs the curl commands for missing images
    """
    print("{} commands to run".format(len(missing_fits_commands)))
    for comm in missing_fits_commands:
        print(comm)
        proc = subprocess.Popen(comm, cwd=fits_dir, shell=True)
        proc.wait()

def find_missing_bad_fits(fits_dir, fits_files, curl_commands):
    """
    find_missing_bad_fits
        returns a list of the missing image download commands
    """
    bad_downloaded_files = {}
    for downloaded_fits in fits_files:
        file_size = os.path.getsize(os.path.join(fits_dir, downloaded_fits))
        file_size_kb = file_size / 1024  # Convert bytes to kilobytes

        if file_size_kb < 34000:
            bad_downloaded_files[downloaded_fits] = file_size_kb

    missing_file_commands = []
    bad_file_sizes = []
    bad_files = []
    for command in curl_commands:
        already_downloaded = False
        for downloaded_fits in fits_files:
            if downloaded_fits in command:
                already_downloaded = True
                if downloaded_fits in bad_downloaded_files:
                    missing_file_commands.append(command)
                    bad_files.append(downloaded_fits)
                    bad_file_sizes.append(bad_downloaded_files[downloaded_fits])
                break

        if not already_downloaded:
            missing_file_commands.append(command)
            bad_files.append(None)
            bad_file_sizes.append(-1)
    
    return missing_file_commands, bad_files, bad_file_sizes

def get_curl_commands(fits_dir, curl_file):
    """
    get_curl_commands
        Gets the list of commands from the download bash file
    """
    curl_file = os.path.join(fits_dir, curl_file)
    with open(curl_file, "r") as filehandle:
        curl_commands = [line.strip() for line in filehandle.readlines()]
    
    if "#!/bin/sh" in curl_commands:
        curl_commands.remove("#!/bin/sh")
    
    return curl_commands

def get_files(fits_dir):
    """
    get_files
        Gets the list of image files in the fits dir
    """
    downloaded_fits = []
    curl_file = None
    for filename in sorted(os.listdir(fits_dir)):
        if ".fits" in filename:
            downloaded_fits.append(filename)
        if "tesscurl" in filename:
            curl_file = filename
    return downloaded_fits, curl_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process FITS files.")
    parser.add_argument("--fits_dir", type=str, default="/cluster/research-groups/hutchinson/data/ml_astro/tess/fits/", 
                        help="Directory containing the FITS files.")
    parser.add_argument("--sectors", type=str, nargs='+', default=["sec_06", "sec_10", "sec_11", "sec_13", "sec_16", "sec_17", "sec_19", "sec_20", "sec_22", "sec_23", "sec_25"], 
                        help="List of sectors to process.")
    
    args = parser.parse_args()
    main(args.fits_dir, args.sectors)
