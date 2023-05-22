import sys
import cv2
import numpy as np

ARGUMENT_COUNT = 7
ARGUMENT_TYPE = ["-input", "-output", "-brightness"]

# Make sure the command line arguments are valid
def validate_argument():
    if len(sys.argv) != ARGUMENT_COUNT:
        return False
    
    command = sys.argv[1::2]
    for index in range(len(command)):
        if command[index] != ARGUMENT_TYPE[index]:
            return False
        
    if int(sys.argv[6]) != float(sys.argv[6]):
        return False
    
    return True

# Save the image
def save_image(processed_image, output_path):
    cv2.imwrite(output_path, processed_image)
    print("Saved successfully!")

# Get the channel number
def get_channel_number(processed_image):
    if len(processed_image.shape) != 3:
        return 1
    else:
        return processed_image.shape[-1]
    
# Function to adjust the image brightness and let the user preview it
# A reasonable brightness factor ranges from (-127, 127)
def adjust_brightness(input_path, output_path, brightness_factor):
    processed_image = cv2.imread(input_path)
    height, width = processed_image.shape[0], processed_image.shape[1]
    channel_number = get_channel_number(processed_image)
    
    for i in range(height):
        for j in range(width):
            for k in range(channel_number):
                processed_image[i, j, k] = np.clip(processed_image[i, j, k] + brightness_factor, 0, 255)
    
    cv2.imshow(output_path, processed_image)
    cv2.waitKey(0)
    save_image(processed_image, output_path)


def main():
    if validate_argument() == False:
        print("ERROR: The command line is invalid!")
        exit()
        
    adjust_brightness(sys.argv[2], sys.argv[4], int(sys.argv[6]))
    print("Adjusted the brightness successfully!")
        
if __name__ == "__main__":
    main()