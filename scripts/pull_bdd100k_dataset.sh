#!/bin/bash

# Function to download, unzip, and cleanup files
download_and_unzip() {
    local urls=("$@")
    local dir=${urls[-1]}
    unset 'urls[-1]'

    # Create the download directory if it doesn't exist
    mkdir -p "$dir"

    # Iterate over the list of URLs
    for url in "${urls[@]}"; do
        # Extract the file name from the URL
        local file_name=$(basename "$url")
        local skip_download=false

        # Check if the URL is for the lane labels and skip if the folder exists
        if [[ "$url" == *"100k_lane_labels_trainval.zip" ]]; then
            if [ -d "$dir/bdd100k/labels" ]; then
                echo "Labels folder already exists. Skipping download of $file_name."
                skip_download=true
            fi
        elif [[ "$url" == *"100k_images_train.zip" ]]; then
            if [ -d "$dir/bdd100k/images/100k/train" ]; then
                echo "Train images folder already exists. Skipping download of $file_name."
                skip_download=true
            fi
        elif [[ "$url" == *"100k_images_val.zip" ]]; then
            if [ -d "$dir/bdd100k/images/100k/val" ]; then
                echo "Validation images folder already exists. Skipping download of $file_name."
                skip_download=true
            fi
        elif [[ "$url" == *"100k_images_test.zip" ]]; then
            if [ -d "$dir/bdd100k/images/100k/test" ]; then
                echo "Test images folder already exists. Skipping download of $file_name."
                skip_download=true
            fi
        fi

        if [ "$skip_download" = false ]; then
            # Download the file
            wget -P "$dir" "$url"

            # Unzip the file
            unzip "$dir/$file_name" -d "$dir"

            # Remove the zip file
            rm "$dir/$file_name"
        fi
    done
}

# List of URLs to download
download_url_list=(
    "https://dl.cv.ethz.ch/bdd100k/data/100k_images_train.zip"
    "https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip"
    "https://dl.cv.ethz.ch/bdd100k/data/100k_images_test.zip"
    "https://dl.cv.ethz.ch/bdd100k/data/100k_lane_labels_trainval.zip"
)

# Ask user for the output directory
read -p "Please enter the output directory for the dataset: " output_dir

# Check if the output directory exists
if [ -d "$output_dir" ]; then
    if [ -d "$output_dir/bdd100k/images/100k" ] || [ -d "$output_dir/bdd100k/labels" ]; then
        echo "BDD100K dataset already exists. Checking for missing parts..."
        # Call the download script and check for missing parts
        download_and_unzip "${download_url_list[@]}" "$output_dir"
    else
        echo "BDD100K dataset does not exist. Downloading..."
        # Call the download script
        download_and_unzip "${download_url_list[@]}" "$output_dir"
    fi
else
    echo "Output directory does not exist. Creating..."
    mkdir -p "$output_dir"
    echo "BDD100K dataset does not exist. Downloading..."
    # Call the download script
    download_and_unzip "${download_url_list[@]}" "$output_dir"
fi

echo "Download, unzip, and cleanup completed."
