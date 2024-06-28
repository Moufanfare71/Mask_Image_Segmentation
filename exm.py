# Now, this function can be called from another script with the 
# appropriate arguments.

# Example usage:

from segment import segment_and_remove_background
segment_and_remove_background(
    'E:/Project/SAM/a1.jpeg',
    'E:/Project/SAM/a1_segmented.jpeg',
    'E:/Project/SAM/a1_mask.jpeg',  # Path for the mask image
    'E:/Project/SAM/a1_masked_area.png',  # Path for the masked area image
    'yolov8n.pt',
    'sam_vit_h_4b8939.pth',
    'vit_h'
)