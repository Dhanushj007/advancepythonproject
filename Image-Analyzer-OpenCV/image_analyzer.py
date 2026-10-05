import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_DIR / "images"
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_FILE = PROJECT_DIR / "image_data.json"


def create_project_folders():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        DATA_FILE.write_text("{}", encoding="utf-8")


def get_image_properties(image):
    image_array = np.asarray(image)
    height, width = image_array.shape[:2]
    channels = 1 if image_array.ndim == 2 else image_array.shape[2]
    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "data_type": str(image_array.dtype),
    }


def choose_image_path():
    selected_path = ""
    try:
        root = tk.Tk()
        root.withdraw()
        selected_path = filedialog.askopenfilename(
            title="Select an image",
            initialdir=str(IMAGES_DIR),
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
    except tk.TclError as error:
        print(f"Could not open the file dialog: {error}")

    if not selected_path:
        selected_path = input(
            "Enter the image path (or press Enter to cancel): "
        ).strip().strip('"')
    return Path(selected_path).expanduser() if selected_path else None


def save_image_data(image_path, image):
    data = {
        "filename": image_path.name,
        **get_image_properties(image),
    }
    try:
        with DATA_FILE.open("w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)
    except OSError as error:
        print(f"Could not save image details to {DATA_FILE}: {error}")
        return
    except (TypeError, ValueError) as error:
        print(f"Could not create image details: {error}")
        return
    print(f"Image details saved to: {DATA_FILE}")


def read_image():
    image_path = choose_image_path()
    if image_path is None:
        print("Image selection cancelled.")
        return None, None
    if not image_path.is_file():
        print(f"Image file not found: {image_path}")
        return None, None

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        print(f"Could not read the image: {error}")
        return None, None
    if image is None:
        print(f"Could not load this image. Check the file: {image_path}")
        return None, None

    save_image_data(image_path, image)
    print(f"Image loaded successfully: {image_path}")
    return image, image_path


def display_image_properties(image):
    if image is None:
        print("Read an image first (menu option 1).")
        return
    properties = get_image_properties(image)
    print("\nImage Properties")
    print(f"Width      : {properties['width']}")
    print(f"Height     : {properties['height']}")
    print(f"Channels   : {properties['channels']}")
    print(f"Data Type  : {properties['data_type']}")


def display_image(image, title="Selected Image"):
    if image is None:
        print("Read an image first (menu option 1).")
        return
    try:
        cv2.imshow(title, image)
        print("Image window opened. Press any key in the image window to continue.")
        cv2.waitKey(0)
    except cv2.error as error:
        print(f"Could not display the image: {error}")
    finally:
        cv2.destroyAllWindows()


def get_integer(prompt):
    try:
        return int(input(prompt))
    except ValueError:
        print("Please enter a whole number.")
        return None


def save_processed_image(image, output_path):
    image_to_save = image
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        if image.ndim == 3 and image.shape[2] == 4:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    try:
        saved = cv2.imwrite(str(output_path), image_to_save)
    except (cv2.error, OSError) as error:
        print(f"Could not save the image to {output_path}: {error}")
        return False
    if not saved:
        print(f"Could not save the image to {output_path}.")
        return False
    print(f"Image saved to: {output_path}")
    return True


def crop_image(image, processed_images):
    if image is None:
        print("Read an image first (menu option 1).")
        return
    start_x = get_integer("Start X: ")
    start_y = get_integer("Start Y: ")
    end_x = get_integer("End X: ")
    end_y = get_integer("End Y: ")
    if None in (start_x, start_y, end_x, end_y):
        return

    height, width = image.shape[:2]
    if not (0 <= start_x < end_x <= width and 0 <= start_y < end_y <= height):
        print(
            f"Invalid crop area. Use X values from 0 to {width} "
            f"and Y values from 0 to {height}."
        )
        return

    cropped = image[start_y:end_y, start_x:end_x]
    path = OUTPUT_DIR / "cropped_image.jpg"
    if save_processed_image(cropped, path):
        processed_images["cropped"] = cropped
        print("Image cropped successfully.")


def resize_image(image, processed_images):
    if image is None:
        print("Read an image first (menu option 1).")
        return
    new_width = get_integer("New width: ")
    new_height = get_integer("New height: ")
    if new_width is None or new_height is None:
        return
    if new_width <= 0 or new_height <= 0:
        print("Width and height must be greater than zero.")
        return

    try:
        resized = cv2.resize(image, (new_width, new_height))
    except cv2.error as error:
        print(f"Could not resize the image: {error}")
        return
    path = OUTPUT_DIR / "resized_image.jpg"
    if save_processed_image(resized, path):
        processed_images["resized"] = resized
        print("Image resized successfully.")


def convert_to_grayscale(image, processed_images):
    if image is None:
        print("Read an image first (menu option 1).")
        return
    try:
        if image.ndim == 2:
            grayscale = image.copy()
        elif image.shape[2] == 4:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error as error:
        print(f"Could not convert the image to grayscale: {error}")
        return

    path = OUTPUT_DIR / "grayscale_image.jpg"
    if save_processed_image(grayscale, path):
        processed_images["grayscale"] = grayscale
        display_image(grayscale, "Grayscale Image")


def save_processed_image_menu(processed_images):
    if not processed_images:
        print(
            "There are no processed images to save. "
            "Crop, resize, or convert an image first."
        )
        return

    names = list(processed_images)
    print("\nProcessed images:")
    for index, name in enumerate(names, start=1):
        print(f"{index}. {name}")
    choice = get_integer("Choose an image to save: ")
    if choice is None or not 1 <= choice <= len(names):
        print("Invalid processed-image choice.")
        return

    name = names[choice - 1]
    default_name = f"{name}_image.jpg"
    filename = input(
        f"Output filename inside the output folder [{default_name}]: "
    ).strip()
    if not filename:
        filename = default_name
    if Path(filename).name != filename or not Path(filename).suffix:
        print("Enter a filename with an extension, not a folder path.")
        return
    save_processed_image(processed_images[name], OUTPUT_DIR / filename)


def view_image_data():
    if not DATA_FILE.is_file():
        print(f"Image data file not found: {DATA_FILE}")
        return
    try:
        data = pd.read_json(DATA_FILE, typ="series")
        if data.empty:
            print("No image details are stored yet. Read an image first.")
            return
        table = pd.DataFrame([data.to_dict()])
        print("\nImage Information:")
        print(table.to_string(index=False))
    except (OSError, ValueError, TypeError) as error:
        print(f"Could not read image data from {DATA_FILE}: {error}")


def generate_graph(image):
    if image is None:
        print("Read an image first (menu option 1).")
        return
    properties = get_image_properties(image)
    labels = ["Width", "Height", "Channels"]
    values = [
        properties["width"],
        properties["height"],
        properties["channels"],
    ]
    graph_path = OUTPUT_DIR / "image_properties_graph.png"
    try:
        plt.figure(figsize=(7, 5))
        plt.bar(labels, values, color=["steelblue", "orange", "seagreen"])
        plt.title("Image Properties")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(graph_path)
        print(f"Graph saved to: {graph_path}")
        plt.show()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Could not generate the graph: {error}")
    finally:
        plt.close()


def display_menu():
    print("\n" + "=" * 32)
    print("IMAGE ANALYZER")
    print("=" * 32)
    print("1. Read Image")
    print("2. Display Image Properties")
    print("3. Display Image")
    print("4. Crop Image")
    print("5. Resize Image")
    print("6. Convert to Grayscale")
    print("7. Save Processed Image")
    print("8. View Image Data")
    print("9. Generate Graph")
    print("10. Exit")


def main():
    try:
        create_project_folders()
    except OSError as error:
        print(f"Could not create project folders or data file: {error}")
        return

    image = None
    image_path = None
    processed_images = {}

    while True:
        display_menu()
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            new_image, new_image_path = read_image()
            if new_image is not None:
                image = new_image
                image_path = new_image_path
                processed_images.clear()
        elif choice == "2":
            display_image_properties(image)
        elif choice == "3":
            title = image_path.name if image_path is not None else "Selected Image"
            display_image(image, title)
        elif choice == "4":
            crop_image(image, processed_images)
        elif choice == "5":
            resize_image(image, processed_images)
        elif choice == "6":
            convert_to_grayscale(image, processed_images)
        elif choice == "7":
            save_processed_image_menu(processed_images)
        elif choice == "8":
            view_image_data()
        elif choice == "9":
            generate_graph(image)
        elif choice == "10":
            print("Exiting Image Analyzer. Goodbye!")
            break
        else:
            print("Invalid menu choice. Enter a number from 1 to 10.")


if __name__ == "__main__":
    main()
