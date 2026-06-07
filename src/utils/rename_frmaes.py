import os

def rename_images(folder_path):
    # Supported image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp','heic')

    # Get all files in the folder and filter for images
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    images.sort()  # Optional: sort for predictable order

    for idx, filename in enumerate(images, start=1):
        file_ext = os.path.splitext(filename)[1]
        new_name = f"ir_13_img_{idx}{file_ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_name}'")

# Example usage:
rename_images("C://Users//Gowtham C K//Desktop//FYP//Green Annotations Datasets")
