import ultralytics
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def predict(image_path, model):
    # Perform object detection using YOLO
    results = model.predict(image_path, save=True)
    for result in results:
        boxes = result.boxes
    bbox = boxes.xyxy.tolist()[0]  # Detect the first bounding box

    print("RESULT FROM YOLO: ", bbox)
    return bbox

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def process_img(image_path, bbox, predictor, output_path_segmented):
    # Read and preprocess the image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    input_box = np.array(bbox)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    # Show mask and bounding box
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')
    plt.savefig(output_path_segmented, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

    print(f"Segmented image saved to {output_path_segmented}")

def remove_background_and_display(image_path, bbox, predictor, output_path_mask, output_path_masked_image):
    # Read and preprocess the image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    input_box = np.array(bbox)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    # Extract segmentation mask from masks
    segmentation_mask = masks[0]

    # Convert segmentation mask to binary mask
    binary_mask = (segmentation_mask > 0.5).astype(np.uint8)

    # Save the binary mask image
    mask_image = (binary_mask * 255).astype(np.uint8)
    cv2.imwrite(output_path_mask, mask_image)

    # # Apply the mask to the original image
    # masked_image = cv2.bitwise_and(image, image, mask=binary_mask)

    # # Save the masked image
    # cv2.imwrite(output_path_masked_image, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

    # Create an empty image with the same dimensions and an alpha channel
    rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    
    # Set the alpha channel based on the binary mask
    rgba_image[:, :, 3] = binary_mask * 255

    # Convert to PIL Image to save with transparency
    pil_image = Image.fromarray(rgba_image)
    pil_image.save(output_path_masked_image)

    print(f"Mask image saved to {output_path_mask}")
    print(f"Masked area image saved to {output_path_masked_image}")

def segment_and_remove_background(image_path, output_path_segmented, output_path_mask, output_path_masked_image, yolo_model_path, sam_checkpoint_path, sam_model_type):
    model = YOLO(yolo_model_path)

    # Run prediction
    bbox = predict(image_path, model)

    # Load SAM model
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
    predictor = SamPredictor(sam)

    # Process and save the segmented image
    process_img(image_path, bbox, predictor, output_path_segmented)

    # Save the mask and masked area image
    remove_background_and_display(image_path, bbox, predictor, output_path_mask, output_path_masked_image)

# Call the function with the correct arguments

