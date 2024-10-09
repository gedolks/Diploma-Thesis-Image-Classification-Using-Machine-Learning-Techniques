# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import shutil

# Load an image in grayscale mode
img = cv2.imread("test_galaxy.jpg", 0)

# Check if the image was loaded correctly, otherwise exit
if img is None:
    print("Error loading image")
    exit()

# Print the dimensions of the image
print(img.shape)

# Get the height and width of the image
height, width = img.shape

# Create an array to hold Gaussian noise
gauss_noise = np.zeros((height, width), dtype=np.uint8)
cv2.randn(gauss_noise, 128, 20)  # Fill the array with random numbers
gauss_noise = (gauss_noise * 0.5).astype(np.uint8)  # Scale the noise intensity

# Add Gaussian noise to the original image
gn_img = cv2.add(img, gauss_noise)

# Prepare a figure for displaying images
fig = plt.figure(dpi=300)

# Display the original image
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

# Display the Gaussian noise
fig.add_subplot(1, 3, 2)
plt.imshow(gauss_noise, cmap='gray')
plt.axis("off")
plt.title("Gaussian Noise")

# Display the image with Gaussian noise added
fig.add_subplot(1, 3, 3)
plt.imshow(gn_img, cmap='gray')
plt.axis("off")
plt.title("Combined")

# Show the figure with the images
plt.show()

# Repeat the process for Uniform noise
uni_noise = np.zeros((height, width), dtype=np.uint8)
cv2.randu(uni_noise, 0, 255)
uni_noise = (uni_noise * 0.5).astype(np.uint8)

# Add uniform noise to the original image
un_img = cv2.add(img, uni_noise)

# Prepare another figure for displaying the results
fig = plt.figure(dpi=300)

# Display the original image
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

# Display the uniform noise
fig.add_subplot(1, 3, 2)
plt.imshow(uni_noise, cmap='gray')
plt.axis("off")
plt.title("Uniform Noise")

# Display the image with uniform noise added
fig.add_subplot(1, 3, 3)
plt.imshow(un_img, cmap='gray')
plt.axis("off")
plt.title("Combined")

# Show the figure with the images
plt.show()

# Repeat the process for Impulse noise
imp_noise = np.zeros((height, width), dtype=np.uint8)
cv2.randu(imp_noise, 0, 255)
imp_noise = cv2.threshold(imp_noise, 245, 255, cv2.THRESH_BINARY)[1]

# Add impulse noise to the original image
in_img = cv2.add(img, imp_noise)

# Prepare another figure for displaying the results
fig = plt.figure(dpi=300)

# Display the original image
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

# Display the impulse noise
fig.add_subplot(1, 3, 2)
plt.imshow(imp_noise, cmap='gray')
plt.axis("off")
plt.title("Impulse Noise")

# Display the image with impulse noise added
fig.add_subplot(1, 3, 3)
plt.imshow(in_img, cmap='gray')
plt.axis("off")
plt.title("Combined")

# Show the figure with the images
plt.show()

# Apply non-local means denoising to the Gaussian noise added image
denoised1 = cv2.fastNlMeansDenoising(gn_img, None, 10, 10)

# Prepare a figure for displaying the denoising results
fig = plt.figure(dpi=300)
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1, 3, 2)
plt.imshow(gn_img, cmap='gray')
plt.axis("off")
plt.title("with Gaussian Noise")

fig.add_subplot(1, 3, 3)
plt.imshow(denoised1, cmap='gray')
plt.axis("off")
plt.title("After Denoising")

# Show the figure with the denoising results
plt.show()

# Repeat denoising for uniform noise added image
denoised2 = cv2.fastNlMeansDenoising(un_img, None, 10, 10)

# Prepare another figure for denoising results
fig = plt.figure(dpi=300)
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1, 3, 2)
plt.imshow(un_img, cmap='gray')
plt.axis("off")
plt.title("with Uniform Noise")

fig.add_subplot(1, 3, 3)
plt.imshow(denoised2, cmap='gray')
plt.axis("off")
plt.title("After Denoising")

# Show the figure with the denoising results
plt.show()

# Repeat denoising for impulse noise added image
denoised3 = cv2.fastNlMeansDenoising(in_img, None, 10, 10)

# Prepare another figure for denoising results
fig = plt.figure(dpi=300)
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1, 3, 2)
plt.imshow(in_img, cmap='gray')
plt.axis("off")
plt.title("with Impulse Noise")

fig.add_subplot(1, 3, 3)
plt.imshow(denoised3, cmap='gray')
plt.axis("off")
plt.title("After Denoising")

# Show the figure with the denoising results
plt.show()

# Apply a median filter to the Gaussian noise added image
blurred1 = cv2.medianBlur(gn_img, 3)

# Prepare a figure to display the filtering results
fig = plt.figure(dpi=300)
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1, 3, 2)
plt.imshow(gn_img, cmap='gray')
plt.axis("off")
plt.title("with Gaussian Noise")

fig.add_subplot(1, 3, 3)
plt.imshow(blurred1, cmap='gray')
plt.axis("off")
plt.title("Median Filter")

# Show the figure with the filtering results
plt.show()

# Repeat median filtering for uniform noise added image
blurred2 = cv2.medianBlur(un_img, 3)

# Prepare another figure to display the filtering results
fig = plt.figure(dpi=300)
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1, 3, 2)
plt.imshow(un_img, cmap='gray')
plt.axis("off")
plt.title("with Uniform Noise")

fig.add_subplot(1, 3, 3)
plt.imshow(blurred2, cmap='gray')
plt.axis("off")
plt.title("Median Filter")

# Show the figure with the filtering results
plt.show()

# Repeat median filtering for impulse noise added image
blurred3 = cv2.medianBlur(in_img, 3)

# Prepare another figure to display the filtering results
fig = plt.figure(dpi=300)
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1, 3, 2)
plt.imshow(in_img, cmap='gray')
plt.axis("off")
plt.title("with Impulse Noise")

fig.add_subplot(1, 3, 3)
plt.imshow(blurred3, cmap='gray')
plt.axis("off")
plt.title("Median Filter")

# Show the figure with the filtering results
plt.show()

# Apply a Gaussian filter to the Gaussian noise added image
blurred21 = cv2.GaussianBlur(gn_img, (3, 3), 0)

# Prepare a figure to display the Gaussian filtering results
fig = plt.figure(dpi=300)
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1, 3, 2)
plt.imshow(gn_img, cmap='gray')
plt.axis("off")
plt.title("with Gaussian Noise")

fig.add_subplot(1, 3, 3)
plt.imshow(blurred21, cmap='gray')
plt.axis("off")
plt.title("Gaussian Filter")

# Show the figure with the Gaussian filtering results
plt.show()

# Repeat Gaussian filtering for uniform noise added image
blurred22 = cv2.GaussianBlur(un_img, (3, 3), 0)

# Prepare another figure to display the Gaussian filtering results
fig = plt.figure(dpi=300)
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1, 3, 2)
plt.imshow(un_img, cmap='gray')
plt.axis("off")
plt.title("with Uniform Noise")

fig.add_subplot(1, 3, 3)
plt.imshow(blurred22, cmap='gray')
plt.axis("off")
plt.title("Gaussian Filter")

# Show the figure with the Gaussian filtering results
plt.show()

# Repeat Gaussian filtering for impulse noise added image
blurred23 = cv2.GaussianBlur(in_img, (3, 3), 0)

# Prepare another figure to display the Gaussian filtering results
fig = plt.figure(dpi=300)
fig.add_subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1, 3, 2)
plt.imshow(in_img, cmap='gray')
plt.axis("off")
plt.title("with Impulse Noise")

fig.add_subplot(1, 3, 3)
plt.imshow(blurred23, cmap='gray')
plt.axis("off")
plt.title("Gaussian Filter")

# Show the figure with the Gaussian filtering results
plt.show()

# Define a function to add noise and denoise an image
def add_noise_and_denoise(image_path, output_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Check if the image was loaded correctly
    if img is None:
        print("Error loading image:", image_path)
        return

    # Get dimensions of the image
    height, width = img.shape
    # Randomly choose a type of noise to add
    noise_type = random.choice(['gaussian', 'uniform', 'impulse'])
    
    # Depending on the chosen noise type, add the noise and apply denoising
    if noise_type == 'gaussian':
        gauss_noise = np.random.normal(128, 20, (height, width)).astype(np.uint8)
        noise_img = cv2.add(img, (gauss_noise * 0.5).astype(np.uint8))
        denoised_img = cv2.GaussianBlur(noise_img, (3, 3), 0)
    elif noise_type == 'uniform':
        uni_noise = np.random.uniform(0, 255, (height, width)).astype(np.uint8)
        noise_img = cv2.add(img, (uni_noise * 0.5).astype(np.uint8))
        denoised_img = cv2.medianBlur(noise_img, 3)
    elif noise_type == 'impulse':
        imp_noise = np.random.uniform(0, 255, (height, width)).astype(np.uint8)
        _, imp_noise = cv2.threshold(imp_noise, 245, 255, cv2.THRESH_BINARY)
        noise_img = cv2.add(img, imp_noise)
        denoised_img = cv2.medianBlur(noise_img, 3)

    # Save the denoised image to the specified output path
    cv2.imwrite(output_path, denoised_img)
    print(f"Processed and saved: {output_path}")

# Define a function to process folders containing images
def process_folders(base_path):
    # Determine the parent directory of the base path
    parent_directory = os.path.abspath(os.path.join(base_path, os.pardir))
    # Define the output directory
    output_base = os.path.join(parent_directory, "training_2")
    
    # Check if the directory exists, and if so, remove it
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
        print(f"Deleted existing directory: {output_base}")

    # Create the directory again
    os.makedirs(output_base)
    print(f"Created new directory: {output_base}")

    # Process each folder in the base path
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            output_folder = os.path.join(output_base, folder)
            os.makedirs(output_folder)

            # Process each image file in the folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    output_path = os.path.join(output_folder, filename)
                    add_noise_and_denoise(image_path, output_path)

# Example usage
base_path = "ALL_IMAGES\\training"
process_folders(base_path)
