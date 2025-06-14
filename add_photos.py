import pandas as pd
import os
import random
from pathlib import Path

def get_all_photos():
    """Get a list of all photos from the photos directory and its subdirectories."""
    photos = []
    photos_dir = Path('photos')
    
    # Walk through all subdirectories
    for root, _, files in os.walk(photos_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Get relative path from photos directory
                rel_path = os.path.relpath(os.path.join(root, file), start='photos')
                photos.append(rel_path)
    
    return photos

def add_photos_to_csv(csv_file, photos_list):
    """Add random photo links to a CSV file."""
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Add a new column for photo URLs
    df['photo_url'] = [f'/photos/{random.choice(photos_list)}' for _ in range(len(df))]
    
    # Save the updated CSV
    output_file = f"{os.path.splitext(csv_file)[0]}_with_photos.csv"
    df.to_csv(output_file, index=False)
    print(f"Updated {csv_file} -> {output_file}")

def main():
    # Get list of all photos
    photos = get_all_photos()
    if not photos:
        print("No photos found in the photos directory!")
        return
    
    print(f"Found {len(photos)} photos")
    
    # Process each CSV file
    csv_files = ['brooklyn.csv', 'manhattan.csv', 'queens.csv']
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            add_photos_to_csv(csv_file, photos)
        else:
            print(f"Warning: {csv_file} not found")

if __name__ == "__main__":
    main() 