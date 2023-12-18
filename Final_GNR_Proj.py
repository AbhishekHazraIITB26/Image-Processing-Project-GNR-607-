import PySimpleGUI as sg
import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, target_size):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_size)
    new_height = int(target_size / aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def apply_filters(image_path, gaussian_kernel_size, simple_average_kernel_size, target_size, variance):
    original_image = cv2.imread(image_path)
    original_image_resized = resize_image(original_image, target_size)

    gaussian_kernel_size = max(3, gaussian_kernel_size // 2 * 2 + 1)
    simple_average_kernel_size = max(3, simple_average_kernel_size // 2 * 2 + 1)

    # Calculate sigma from the provided variance
    sigma = np.sqrt(variance)

    gaussian_filtered_image = cv2.GaussianBlur(original_image_resized, (gaussian_kernel_size, gaussian_kernel_size), sigma)
    kernel = np.ones((simple_average_kernel_size, simple_average_kernel_size), np.float32) / (simple_average_kernel_size**2)
    simple_average_filtered_image = cv2.filter2D(original_image_resized, -1, kernel)

    return original_image_resized, simple_average_filtered_image, gaussian_filtered_image

def edge_detection(image):
    # Apply Sobel operator for edge detection
    edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    return edges

def create_figure(image, title):
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='jet')
    ax.set_title(title)
    ax.axis('off')
    return fig, im

def plot_histograms(original_image, simple_average_filtered_image, gaussian_filtered_image):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.hist(original_image.flatten(), bins=256, color='blue', alpha=0.7, label='Original Image')
    ax.hist(simple_average_filtered_image.flatten(), bins=256, color='green', alpha=0.5, label='Simple Average Filtered')
    ax.hist(gaussian_filtered_image.flatten(), bins=256, color='red', alpha=0.3, label='Gaussian Filtered')

    ax.legend()
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Pixel Intensity')

    plt.tight_layout()
    return fig

def main():
    layout = [
        [sg.Text('Select an image:'), sg.InputText(key='image_path'), sg.FileBrowse()],
        [sg.Text('Gaussian Blur Kernel Size (Odd):'), sg.Slider((3, 15), 3, orientation='h', key='gaussian_kernel_size')],
        [sg.Text('Simple Averaging Kernel Size (Odd):'), sg.Slider((3, 15), 3, orientation='h', key='simple_average_kernel_size')],
        [sg.Text('Image Size:'), sg.Slider((100, 800), 300, orientation='h', key='target_size')],
        [sg.Text('Variance'), sg.Slider((1, 100000), 300, orientation='h', key='variance')],
        [sg.Button('Apply Filters'), sg.Button('Show Histograms'), sg.Button('Show Edges'), sg.Button('Exit')],
        [sg.Text('Original Image', size=(15, 1), justification='center'),
        sg.Image(key='original_image', size=(300, 300), tooltip='Original Image'),
        sg.Text('Simple Filter', size=(15, 1), justification='center'),
        sg.Image(key='simple_average_filtered_image', size=(300, 300), tooltip='Simple Filtered Image'),
        sg.Text('Gaussian Smoothed', size=(15, 1), justification='center'),
        sg.Image(key='gaussian_filtered_image', size=(300, 300), tooltip='Gaussian Smoothed Image')]
    ]

    window = sg.Window('Image Filter GUI', layout, finalize=True, resizable=True)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Apply Filters':
            image_path = values['image_path']
            gaussian_kernel_size = int(values['gaussian_kernel_size'])
            simple_average_kernel_size = int(values['simple_average_kernel_size'])
            target_size = int(values['target_size'])
            variance = int(values['variance'])

            if image_path:
                original_image, simple_average_filtered_image, gaussian_filtered_image = apply_filters(image_path, gaussian_kernel_size, simple_average_kernel_size, target_size, variance)

                original_imgbytes = cv2.imencode('.png', original_image)[1].tobytes()
                simple_average_filtered_imgbytes = cv2.imencode('.png', simple_average_filtered_image)[1].tobytes()
                gaussian_filtered_imgbytes = cv2.imencode('.png', gaussian_filtered_image)[1].tobytes()

                window['original_image'].update(data=original_imgbytes)
                window['simple_average_filtered_image'].update(data=simple_average_filtered_imgbytes)
                window['gaussian_filtered_image'].update(data=gaussian_filtered_imgbytes)

        elif event == 'Show Histograms':
            if image_path:
                original_image, simple_average_filtered_image, gaussian_filtered_image = apply_filters(image_path, gaussian_kernel_size, simple_average_kernel_size, target_size, variance)
                fig_histogram = plot_histograms(original_image, simple_average_filtered_image, gaussian_filtered_image)
                plt.show()

        elif event == 'Show Edges':
            if image_path:
                original_image, simple_average_filtered_image, gaussian_filtered_image = apply_filters(image_path, gaussian_kernel_size, simple_average_kernel_size, target_size, variance)
                edges_original = edge_detection(original_image)
                edges_simple_average = edge_detection(simple_average_filtered_image)
                edges_gaussian = edge_detection(gaussian_filtered_image)

                layout_edges = [
                    [sg.Text('Edges (Original)', size=(15, 1), justification='center'),
                    sg.Image(key='edges_original', size=(300, 300), tooltip='Edges (Original)'),
                    sg.Text('Edges (Simple Average)', size=(20, 1), justification='center'),
                    sg.Image(key='edges_simple_average', size=(300, 300), tooltip='Edges (Simple Average)'),
                    sg.Text('Edges (Gaussian)', size=(15, 1), justification='center'),
                    sg.Image(key='edges_gaussian', size=(300, 300), tooltip='Edges (Gaussian)')],
                    [sg.Button('Close Edges Window')]
                ]

                window_edges = sg.Window('Edge Images', layout_edges, finalize=True, resizable=True)

                edges_original_imgbytes = cv2.imencode('.png', edges_original)[1].tobytes()
                edges_simple_average_imgbytes = cv2.imencode('.png', edges_simple_average)[1].tobytes()
                edges_gaussian_imgbytes = cv2.imencode('.png', edges_gaussian)[1].tobytes()

                window_edges['edges_original'].update(data=edges_original_imgbytes)
                window_edges['edges_simple_average'].update(data=edges_simple_average_imgbytes)
                window_edges['edges_gaussian'].update(data=edges_gaussian_imgbytes)

                while True:
                    event_edges, _ = window_edges.read()

                    if event_edges == sg.WIN_CLOSED or event_edges == 'Close Edges Window':
                        window_edges.close()
                        break

    window.close()

if __name__ == '__main__':
    main()
