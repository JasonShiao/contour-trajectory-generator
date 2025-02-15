import cv2
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from collections import Counter



def sort_contour_points(contour):
    sorted_contour = [contour[0]]
    remaining_points = contour[1:]

    while len(remaining_points) > 0:
        last_point = sorted_contour[-1]
        # Find the nearest neighbor
        distances = np.linalg.norm(remaining_points - last_point, axis=1)
        nearest_idx = np.argmin(distances)
        sorted_contour.append(remaining_points[nearest_idx])
        # Remove the nearest neighbor from the remaining points
        remaining_points = np.delete(remaining_points, nearest_idx, axis=0)

    # Ensure the contour is closed by appending the starting point at the end
    sorted_contour.append(sorted_contour[0])
    return np.array(sorted_contour)


from matplotlib.animation import FuncAnimation
def animate_contour(contour, title=None):
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    line, = ax.plot([], [], lw=2, label="Parametric Curve")
    point, = ax.plot([], [], 'ro', label="Current Point")
    # Find the min and max of x and y for the axis limits
    ax_min = min(min(contour[:, 0]), min(contour[:, 1]))
    ax_max = max(contour[:, 0].max(), contour[:, 1].max())
    margin = 0.05  # 5% margin
    x_range = ax.set_xlim(ax_min - margin * (ax_max - ax_min), ax_max + margin * (ax_max - ax_min))
    y_range = ax.set_ylim(ax_min - margin * (ax_max - ax_min), ax_max + margin * (ax_max - ax_min))
    print(f"x_range: {x_range}, y_range: {y_range}")
    #margin = 50
    #ax.set_xlim(min_x - margin, max_x + margin)
    #ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title('Parametric Curve Animation')

    # Update function for the animation
    def update(frame):
        # Update the line and point
        #line.set_data(xd[:frame], yd[:frame])
        point.set_data([contour[:frame, 0]], [contour[:frame, 1]])
        return line, point

    # Create the animation
    num_frames = len(contour)
    ani = FuncAnimation(fig, update, frames=num_frames, interval=30, blit=True)
    # Show the animation
    plt.show()


def connect_nearby_endpoints(edges, max_dist=10):
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a blank canvas for the new connections
    connections = np.zeros_like(edges)

    # Collect the endpoints of all contours
    endpoints = []
    for contour in contours:
        if len(contour) > 1:  # Avoid single-point contours
            endpoints.append(contour[0][0])  # Start point
            endpoints.append(contour[-1][0])  # End point

    # Convert endpoints to a NumPy array for easier processing
    endpoints = np.array(endpoints)

    # Iterate through endpoints to find and connect nearby points
    for i, pt1 in enumerate(endpoints):
        for j, pt2 in enumerate(endpoints):
            if i < j:  # Avoid duplicate checks
                dist = np.linalg.norm(pt1 - pt2)
                if dist < max_dist:
                    # Draw a line between close endpoints
                    cv2.line(connections, tuple(pt1), tuple(pt2), 255, 1)

    # Combine the original edges with the new connections
    connected_edges = cv2.bitwise_or(edges, connections)

    return connected_edges


def parametric_curve_generator(img_filepath, desired_size, debug=False):
    # Load image
    img = cv2.imread(img_filepath)
    
    # Preprocess: Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Binarize the image
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations = 1)
    
    height, width = img.shape
    new_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Preprocess to smooth the edges
    # Method 1: Canny Edge Detection
    edges = cv2.Canny(img, 100, 200)
    
    if debug:
        cv2.imshow('Edges', edges)
        cv2.waitKey(0)

    edges = connect_nearby_endpoints(edges, max_dist=20)

    if debug:
        cv2.imshow('Connected Edges', edges)
        cv2.waitKey(0)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours found:", len(contours))

    # Extract points from the max contour
    contour = max(contours, key=cv2.contourArea).squeeze()

    # Get bounding box of the contour
    bound_x, bound_y, bound_w, bound_h = cv2.boundingRect(contour)
    offset_x = bound_x + bound_w // 2
    offset_y = bound_y + bound_h // 2
    rescale = desired_size / max(bound_w, bound_h)

    # Ensure the contour is closed
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    if debug:
        animate_contour(contour)

    # Resample the contour to have equidistant points for uniform representation:
    from scipy.interpolate import splprep, splev
    tck, u = splprep([contour[:, 0], contour[:, 1]], s=0)
    resampled = splev(np.linspace(0, 1, 500), tck)
    contour = np.vstack(resampled).T

    # The resulting contour points are not sorted, so sort them to make a continuous path
    contour = sort_contour_points(contour)
    if debug:
        animate_contour(contour)

    # Plot the contour points
    x_points, y_points = contour[:, 0], contour[:, 1]
    plt.figure(figsize=(8, 8))
    plt.scatter(x_points, y_points, s=10, color='red', label='Contour Points')
    plt.axis('equal')
    plt.title('Contour Points')
    plt.legend()
    plt.show()

    if debug:
        cv2.drawContours(new_image, [max(contours, key=cv2.contourArea)], -1, (127, 30, 255), 3)
        cv2.imshow('Max Contour', new_image)
        cv2.waitKey(0)
        cv2.drawContours(new_image, contours, -1, (127, 30, 255), 3)
        cv2.imshow('Contours', new_image)
        cv2.waitKey(0)

    # Convert the contour to complex numbers
    complex_contour = contour[:, 0] + 1j * contour[:, 1]

    fourier_coeffs = np.fft.fft(complex_contour)
    num_coeffs = 100  # Adjust for accuracy vs simplicity
    
    # For debug only, reconstruct the contour from the reduced coefficients
    #reduced_coeffs = np.zeros_like(fourier_coeffs)
    #reduced_coeffs[:num_coeffs] = fourier_coeffs[:num_coeffs]
    #reconstructed_contour = np.fft.ifft(reduced_coeffs)
    #x_original, y_original = max_contour[:, 0, 0], max_contour[:, 0, 1]
    #x_reconstructed, y_reconstructed = reconstructed_contour.real, reconstructed_contour.imag
    #
    #if debug:
    #    plt.figure(figsize=(8, 8))
    #    #plt.plot(x_original, y_original, label='Original Contour', alpha=0.6)
    #    plt.plot(x_reconstructed, y_reconstructed, label=f'Reconstructed with {num_coeffs} Coeffs', linewidth=2)
    #    plt.legend()
    #    plt.axis('equal')
    #    plt.title('Fourier Silhouette Approximation')
    #    plt.show()
    
    # Create a mask to retain only the desired number of coefficients
    if debug:
        #mask = np.zeros_like(fourier_coeffs, dtype=bool)
        #mask[:num_coeffs] = True

        # Apply the mask to get reduced coefficients
        #reduced_coeffs = np.where(mask, fourier_coeffs, 0)

        # Compute the corresponding frequencies
        freq = np.fft.fftfreq(len(fourier_coeffs), d=0.1)

        # Filter the frequency, magnitude, and phase arrays based on the mask
        #filtered_freq = freq[mask]
        magnitude = np.abs(fourier_coeffs)
        phase = np.angle(fourier_coeffs)

        # Plotting
        plt.figure(figsize=(12, 6))

        # Plot magnitude spectrum
        plt.subplot(2, 1, 1)
        plt.plot(np.fft.fftshift(freq),np.fft.fftshift(magnitude))
        plt.title('Magnitude Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid()

        # Plot phase spectrum
        plt.subplot(2, 1, 2)
        plt.plot(np.fft.fftshift(freq), np.fft.fftshift(phase))
        plt.title('Phase Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (radians)')
        plt.grid()

        plt.tight_layout()
        plt.show()

    # Generate sinusoidal parametric equations from the Fourier series
    t = sp.Symbol('t')
    x_t = 0
    y_t = 0

    # Select the lowest n frequencies (-n/2, n/2) to approximate the contour
    for k in range(-num_coeffs // 2, num_coeffs // 2):
        # Handle negative frequencies with symmetry
        index = k % len(fourier_coeffs)
        coeff = fourier_coeffs[index]
        
        # Extract amplitude, phase, and frequency
        amplitude = abs(coeff) / len(complex_contour)
        phase = np.angle(coeff)
        frequency = k
        
        # Add the sinusoidal term to x(t) and y(t)
        x_t += amplitude * sp.cos(2 * sp.pi * frequency * t + phase)
        y_t += amplitude * sp.sin(2 * sp.pi * frequency * t + phase)


    x_t = (x_t - offset_x) * rescale
    y_t = (y_t - offset_y) * rescale
    d_x_t = sp.diff(x_t, t)
    d_y_t = sp.diff(y_t, t)
    
    x_func = sp.lambdify(t, x_t, modules='numpy')
    y_func = sp.lambdify(t, y_t, modules='numpy')
    d_x_func = sp.lambdify(t, d_x_t, modules='numpy')
    d_y_func = sp.lambdify(t, d_y_t, modules='numpy')
    
    return x_func, y_func, d_x_func, d_y_func
