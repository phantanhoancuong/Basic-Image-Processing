import sys
import cv2
import numpy as np

ARGUMENT_COUNT = 7
ARGUMENT_TYPE = ["-input", "-output", "-contrast"]


# Make sure the command line arguments are valid
def validate_argument():
    if len(sys.argv) != ARGUMENT_COUNT:
        return False
    
    command = sys.argv[1::2]
    for index in range(len(command)):
        if command[index] != ARGUMENT_TYPE[index]:
            return False
    
    if float(sys.argv[6]) < 0:
        return False
    
    return True


# Save the image
def save_image(input_image, output_path):
    cv2.imwrite(output_path, input_image)
    print("Saved successfully!")

# Get the channel number
def get_channel_number(processed_image):
    if len(processed_image.shape) != 3:
        return 1
    else:
        return processed_image.shape[-1]

# Function to adjust the image contrast
def adjust_contrast(input_path, output_path, contrast_factor):
    processed_image = cv2.imread(input_path)
    height, width = processed_image.shape[0], processed_image.shape[1]
    channel_number = get_channel_number(processed_image)
    
    for i in range(height):
        for j in range(width):
            for k in range(channel_number):
                processed_image[i, j, k] = np.clip(int(processed_image[i, j, k] * contrast_factor), 0, 255)
                
    
    cv2.imshow(output_path, processed_image)
    cv2.waitKey(0)
    save_image(processed_image, output_path)


def main():
    if validate_argument() == False:
        print("ERROR: The command line is invalid!")
        exit()
        
    adjust_contrast(sys.argv[2], sys.argv[4], float(sys.argv[6]))
    print("Adjusted the contrast successfully!")
        
if __name__ == "__main__":
    main()