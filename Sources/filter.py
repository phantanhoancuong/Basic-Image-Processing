import sys
import cv2
import numpy as np
from PIL import Image

ARGUMENT_COUNT = 9
ARGUMENT_TYPE = ["-input", "-output", "-filter", "-size"]
FILTER_TYPE = ["avg", "med", "gau"]

# Make sure the command line arguments are valid
def validate_argument():
    if len(sys.argv) != ARGUMENT_COUNT:
        return False
    
    command = sys.argv[1::2]
    for index in range(len(command)):
        if command[index] != ARGUMENT_TYPE[index]:
            return False
    
    if sys.argv[6] in FILTER_TYPE == False:
        return False

    return True


# Save the image
def save_image(processed_image, output_path):
    cv2.imwrite(output_path, processed_image)
    print("Saved successfully!")


# Get the average kernel for convolution
def get_average_kernel(kernel_size):
    kernel = np.ones((kernel_size, kernel_size))
    kernel = np.multiply(kernel, 1 / (kernel_size ** 2))
    return kernel
    
    
# Boundary padding for median kernel
def padding(processed_image, a):
    padded_image = np.zeros((processed_image.shape[0] + a * 2, processed_image.shape[1] + a * 2))
    padded_image[a:-a,a:-a] = processed_image
    return padded_image

    
# Apply the median kernel
def median_filter(processed_image, kernel_size):
    indexer = kernel_size // 2
    temp = []
    result = np.zeros((len(processed_image), len(processed_image[0])))
    
    for i in range(len(processed_image)):

        for j in range(len(processed_image[0])):

            for z in range(kernel_size):
                if i + z - indexer < 0 or i + z - indexer > len(processed_image) - 1:
                    for c in range(kernel_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(processed_image[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(1, kernel_size):
                            temp.append(processed_image[i + z - indexer][j + k - indexer])

            temp.sort()
            result[i][j] = temp[len(temp) // 2]
            temp = []
    return result


# Get the Gaussian kernel for convolution
def get_gaussian_kernel(kernel_size):
    kernel = []
    sigma = (kernel_size - 1) / 6
    min_value = None
    max_value = None
    origin_offset = (-1) * ((kernel_size - 1) / 2)
    for i in range(kernel_size):
        true_i = i + origin_offset
        row = []
        for j in range(kernel_size):
            true_j = j + origin_offset
            temp = 1/(2 * np.pi * sigma ** 2)
            temp = temp * np.e ** ((-1) * (true_i ** 2 + true_j ** 2)/(2 * sigma ** 2))
            row.append(temp)
            
            if max_value == None:
                max_value = abs(temp)
            else:
                if abs(temp) > max_value:
                    max_value = abs(temp)
                    
            if min_value == None:
                min_value = abs(temp)
            else:
                if abs(temp) < max_value:
                    min_value = abs(temp)
            
        kernel.append(row)
    discrete_factor = round(max_value / min_value)
    kernel = np.multiply(discrete_factor, kernel)
    kernel = np.rint(kernel)
    kernel = np.asarray(kernel)
    return kernel, np.sum(kernel)


# Get the channel number
def get_channel_number(processed_image):
    if len(processed_image.shape) != 3:
        return 1
    else:
        return processed_image.shape[-1]


# Function to apply a convolutional filter to the input image
def apply_filter(input_path, output_path, filter_type, kernel_size):
    processed_image = cv2.imread(input_path)
    
    match filter_type:
        case "avg":
            average_kernel = get_average_kernel(kernel_size)
            processed_image = cv2.filter2D(processed_image, -1, average_kernel)
            shown_image = processed_image
            
        case "med":
            channel_image = []
            for i in range(get_channel_number(processed_image)):
                temp = cv2.split(processed_image)[i]
                temp = median_filter(temp, kernel_size)
                channel_image.append(temp)
                
            median_image = cv2.merge(channel_image)
            median_image = cv2.normalize(median_image, None, 0, 1.0, cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            shown_image = median_image
            processed_image = shown_image * 255
            
        case "gau":
            gaussian_kernel = get_gaussian_kernel(kernel_size)
            processed_image = cv2.filter2D(processed_image, - 1, gaussian_kernel[0] / gaussian_kernel[1])
            shown_image = processed_image

    cv2.imshow(output_path, shown_image)
    cv2.waitKey(0)
    save_image(processed_image, output_path)


def main():
    if validate_argument() == False:
        print("ERROR: The command line is invalid!")
        exit()
        
    apply_filter(sys.argv[2], sys.argv[4], sys.argv[6], int(sys.argv[8]))
    print("Completed the filter application!")
        
if __name__ == "__main__":
    main()