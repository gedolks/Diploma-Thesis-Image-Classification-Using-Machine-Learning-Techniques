#ADD NOISE TO A SINGLE IMAGE, SAME WITH THE test_noise.py but single images

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image in grayscale mode
image_path = "test_galaxy.jpg"
loaded_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded properly, raise an error if not
if loaded_image is None:
    raise ValueError("Image not found")

# Print the dimensions of the loaded image
print("Image dimensions:", loaded_image.shape)
img_height, img_width = loaded_image.shape

# Define a function to add different types of noise to an image
def add_noise(image, noise_type, mean=128, sigma=20, low=0, high=255, thresh=245):
    # Create a zero matrix with the same dimensions as the image
    noise = np.zeros((img_height, img_width), dtype=np.uint8)

    # Apply Gaussian noise
    if noise_type == 'gaussian':
        cv2.randn(noise, mean, sigma)  # Generate random noise
        noise = (noise * 0.5).astype(np.uint8)  # Scale and convert to uint8

    # Apply uniform noise
    elif noise_type == 'uniform':
        cv2.randu(noise, low, high)  # Generate uniform noise
        noise = (noise * 0.5).astype(np.uint8)  # Scale and convert to uint8

    # Apply impulse noise
    elif noise_type == 'impulse':
        cv2.randu(noise, low, high)  # Generate random values
        _, noise = cv2.threshold(noise, thresh, high, cv2.THRESH_BINARY)  # Apply threshold

    return cv2.add(image, noise)  # Add the generated noise to the original image

# Function to display one or more images with titles
def display_images(images, titles):
    for img, title in zip(images, titles):
        plt.figure()  # Create a new figure for each image
        plt.imshow(img, cmap='gray')  # Display image in grayscale
        plt.axis("off")  # Turn off axis
        plt.title(title)  # Set title for the image
        plt.show()  # Display the image

# Generate and display images with different types of noise
gaussian_noise_img = add_noise(loaded_image, 'gaussian')
uniform_noise_img = add_noise(loaded_image, 'uniform')
impulse_noise_img = add_noise(loaded_image, 'impulse')

# Display images with different types of noise
display_images([loaded_image, gaussian_noise_img], ["Original", "Gaussian Noise"])
display_images([loaded_image, uniform_noise_img], ["Original", "Uniform Noise"])
display_images([loaded_image, impulse_noise_img], ["Original", "Impulse Noise"])

# Apply a Non-local Means Denoising algorithm to denoise images
denoise_params = [None, 10, 10]
denoised_gaussian = cv2.fastNlMeansDenoising(gaussian_noise_img, *denoise_params)
denoised_uniform = cv2.fastNlMeansDenoising(uniform_noise_img, *denoise_params)
denoised_impulse = cv2.fastNlMeansDenoising(impulse_noise_img, *denoise_params)

# Display the original, noisy, and denoised images
display_images([loaded_image, gaussian_noise_img, denoised_gaussian], ["Original", "With Gaussian Noise", "Denoised"])
display_images([loaded_image, uniform_noise_img, denoised_uniform], ["Original", "With Uniform Noise", "Denoised"])
display_images([loaded_image, impulse_noise_img, denoised_impulse], ["Original", "With Impulse Noise", "Denoised"])

# Applying and displaying results of Median and Gaussian filters
median_blur = lambda img: cv2.medianBlur(img, 3)  # Define median blur function
gaussian_blur = lambda img: cv2.GaussianBlur(img, (3, 3), 0)  # Define Gaussian blur function

# Display results of Median filtering on noisy images
display_images([loaded_image, gaussian_noise_img, median_blur(gaussian_noise_img)], ["Original", "With Gaussian Noise", "Median Filter"])
display_images([loaded_image, uniform_noise_img, median_blur(uniform_noise_img)], ["Original", "With Uniform Noise", "Median Filter"])
display_images([loaded_image, impulse_noise_img, median_blur(impulse_noise_img)], ["Original", "With Impulse Noise", "Median Filter"])

# Display results of Gaussian filtering on noisy images
display_images([loaded_image, gaussian_noise_img, gaussian_blur(gaussian_noise_img)], ["Original", "With Gaussian Noise", "Gaussian Filter"])
display_images([loaded_image, uniform_noise_img, gaussian_blur(uniform_noise_img)], ["Original", "With Uniform Noise", "Gaussian Filter"])
display_images([loaded_image, impulse_noise_img, gaussian_blur(impulse_noise_img)], ["Original", "With Impulse Noise", "Gaussian Filter"])
