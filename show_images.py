import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_images(folder_path, captions=None, textbox_text=None):
    """
    Display all images in the specified folder_path with optional captions and a textbox.
    """
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out non-image files
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not image_files:
        print("No images found in the folder.")
        return
    
    # Set up the grid for displaying images
    num_images = len(image_files)
    rows = (num_images + 2) // 3
    cols = min(num_images, 3)
    
    # Adjust window size based on the number of images
    figsize = (16, 9)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Create a list of captions if not provided
    if captions is None:
        captions = [f"Image {i+1}" for i in range(num_images)]
    
    # Display each image
    for i, (image_file, caption) in enumerate(zip(image_files, captions)):
        image_path = os.path.join(folder_path, image_file)
        image = mpimg.imread(image_path)
        
        # Calculate the position in the grid
        row = i // 3
        col = i % 3
        
        # Display the image with caption
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.imshow(image)
        ax.set_title(caption)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_images, rows * cols):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')
    
    # Add a textbox
    if textbox_text:
        fig.text(0.5, 0.02, textbox_text, ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Specify the folder path where images are located
folder_path = "demo"

# Specify captions for each image
captions = [
    "First Step: Execute health_care_predictor.py",
    "Upon completion we can run application.py",
    "We can view any patients information and stroke risk",
    "Edit a patients information",
    "View treatment recommendations for that patient",
    "View common info based on the data",
    "Create new patients can also be selected",
    "From here we can input all of a patients information"
]

# Text for the textbox
textbox_text = "Note: health_care_predictor.py must be run before application.py for application to function."

# Display images in the specified folder with captions and textbox
display_images(folder_path, captions, textbox_text)
