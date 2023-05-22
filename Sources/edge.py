import sys
import cv2
import numpy as np

ARGUMENT_COUNT = 9
ARGUMENT_TYPE = ["-input", "-output", "-edge", "-size"]
FILTER_TYPE = ["sobel", "prewitt", "laplace"]

# Make sure the command line arguments are valid
# 1. If the commands are according to the manual
# 2. if the filter type is supported
# 3. If the kernel size is supported
def validate_argument():
    if len(sys.argv) != ARGUMENT_COUNT:
        return False
    
    command = sys.argv[1::2]
    for index in range(len(command)):
        if command[index] != ARGUMENT_TYPE[index]:
            return False
    
    if sys.argv[6] in FILTER_TYPE == False:
        return False
    
    if int(sys.argv[8]) % 2 == 0 or int(sys.argv[8]) <= 1:
        return False

    return True


# Save the image
def save_image(processed_image, output_path):
    cv2.imwrite(output_path, processed_image)
    print("Saved successfully!")

# Calculate the Sobel kernels for convolution
def get_sobel_kernel(kernel_size):
    max_weight = kernel_size * (kernel_size - 1)
    origin_offset = (-1) * ((kernel_size - 1) / 2)
    kernelx = []
    kernely = []
    for i in range(kernel_size):
        true_i = i + origin_offset
        rowx = []
        rowy = []
        for j in range(kernel_size):
            
            true_j = j + origin_offset
            if true_i != 0 and true_j != 0:
                rowx.append(max_weight * (true_i / (true_i ** 2 + true_j ** 2)))
                rowy.append(max_weight * (true_j / (true_i ** 2 + true_j ** 2)))
            else:
                rowx.append(0)
                rowy.append(0)
        kernelx.append(rowx)
        kernely.append(rowy)
    kernelx = np.asarray(kernelx)
    kernely = np.asarray(kernely)
    return kernelx, kernely

# Calculate the Prewitt kernels for convolution
def get_prewitt_kernel(kernel_size):
    kernelx = []
    kernely = []
    for i in range(kernel_size):
        rowy = []
        if i < (kernel_size - 1) / 2:
            rowx = [1] * kernel_size
        elif i == (kernel_size - 1) / 2:
            rowx = [0] * kernel_size
        else:
            rowx = [-1] * kernel_size
            
        for j in range(kernel_size):
            if j < (kernel_size - 1) / 2:
                rowy.append(-1) 
            elif j == (kernel_size - 1) / 2:
                rowy.append(0)    
            else:
                rowy.append(1)
                        
        kernelx.append(rowx)
        kernely.append(rowy)
    kernelx = np.asarray(kernelx)
    kernely = np.asarray(kernely)
    return kernelx, kernely    

# Calculate the Laplacian kernel
def get_laplacian_kernel(kernel_size):
    sigma = (kernel_size - 1) / 6
    origin_offset = (-1) * ((kernel_size - 1) / 2)
    max_value = None
    min_value = None
    
    kernel = []
    for i in range(kernel_size):
        row = []
        true_i = i + origin_offset
        
        for j in range(kernel_size):
            true_j = j + origin_offset
                
            LoG = (-1) / (np.pi * sigma ** 4)
            LoG = LoG * (1 - (true_i ** 2 + true_j ** 2) / (2 * sigma ** 2))
            LoG = LoG * np.e ** (-((true_i ** 2 + true_j ** 2) / (2 * sigma ** 2)))
            
            if max_value == None:
                max_value = abs(LoG)
            else:
                if abs(LoG) > max_value:
                    max_value = abs(LoG)
                    
            if min_value == None:
                min_value = abs(LoG)
            else:
                if abs(LoG) < max_value:
                    min_value = abs(LoG)
                
            row.append(LoG)
        kernel.append(row)
        
    discrete_factor = round(max_value / min_value)
    kernel = np.multiply(discrete_factor, kernel)
    kernel = np.rint(kernel)
    kernel = np.asarray(kernel)
    return kernel

# Function to apply a convolutional filter to the input image
def apply_filter(input_path, output_path, filter_type, kernel_size):
    processed_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    match filter_type:
        case "sobel":
            kernel = get_sobel_kernel(kernel_size)
            imgx = cv2.filter2D(processed_image, -1, kernel[0])
            imgy = cv2.filter2D(processed_image, -1, kernel[1])
            processed_image = cv2.addWeighted(imgx, 0.5, imgy, 0.5, 0)
            
        case "prewitt":
            kernel = get_prewitt_kernel(kernel_size)
            imgx = cv2.filter2D(processed_image, -1, kernel[0])
            imgy = cv2.filter2D(processed_image, -1, kernel[1])
            processed_image = cv2.addWeighted(imgx, 0.5, imgy, 0.5, 0)
            
        case "laplace":
            kernel = get_laplacian_kernel(kernel_size)
            processed_image = cv2.filter2D(processed_image, -1, kernel)
            
    cv2.imshow(output_path, processed_image)
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