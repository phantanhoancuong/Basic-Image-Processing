import sys
import cv2
import numpy as np

ARGUMENT_COUNT = 5
ARGUMENT_TYPE = ["-input", "-output"]

# Make sure the command line arguments are valid
def validate_argument():
    if len(sys.argv) != ARGUMENT_COUNT:
        return False
    
    command = sys.argv[1::2]
    for index in range(len(command)):
        if command[index] != ARGUMENT_TYPE[index]:
            return False
    
    return True

# Save the image
def save_image(input_image, output_path):
    cv2.imwrite(output_path, input_image)
    print("Saved successfully!")


# Function to convert to grayscale and let the user preview it
def rgb_to_grayscale(input_path, output_path):
    input_image = cv2.imread(input_path)
    H, W = input_image.shape[:2]
    output_image = np.zeros((H,W), np.uint8)
    for i in range(H):
        for j in range(W):
            output_image[i, j] = np.clip(0.2126 * input_image[i, j, 0]  + 0.7152 * input_image[i, j, 1] + 0.0722 * input_image[i, j, 2], 0, 255)

    cv2.imshow(output_path, output_image)
    cv2.waitKey(0)
    save_image(output_image, output_path)


def main():
    if validate_argument() == False:
        print("ERROR: The command line is invalid!")
        exit()
    
    rgb_to_grayscale(sys.argv[2], sys.argv[4])
    print("Completed the conversion successfully!")
        
if __name__ == "__main__":
    main()