import cv2
import numpy as np
import os

# Function to colorize the image
def colorize_image(image_path, output_path):
    DIR = r"C:\Users\chauh\Downloads\colorize\Model"
    
    prototxt = os.path.join(DIR, r"D:\project\Model\colorization_deploy_v2.prototxt")
    points = os.path.join(DIR, r"D:\project\Model\pts_in_hull.npy")
    model = os.path.join(DIR, r"D:\project\Model\colorization_release_v2.caffemodel")

    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Load cluster centers
    pts = np.load(points)
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    # Load the black-and-white image
    bw_image = cv2.imread(image_path)
    scaled = bw_image.astype(np.float32) / 255.0

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    L = lab_image[:, :, 0]

    # Resize the L channel to 224x224 (what the network expects)
    L_resized = cv2.resize(L, (224, 224))
    L_resized -= 50  # Subtract 50 to center the L values

    # Set the input to the network
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_resized = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))

    # Combine original L with the predicted ab channels
    lab_output = np.concatenate((L[:, :, np.newaxis], ab_resized), axis=2)

    # Convert LAB back to BGR
    colorized_image = cv2.cvtColor(lab_output, cv2.COLOR_LAB2BGR)
    colorized_image = np.clip(colorized_image, 0, 1)

    # Convert to uint8 and save the image
    colorized_image = (255 * colorized_image).astype(np.uint8)
    cv2.imwrite(output_path, colorized_image)

    return output_path
