import cv2
import streamlit as st
import numpy as np
from PIL import Image
import imutils

st.title("Image Stitching App")

# Add a sidebar
sidebar = st.sidebar
uploaded_files = sidebar.file_uploader("Choose images to process...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Initialize an empty list for storing images
images = []

# Convert uploaded images to OpenCV format and append to list
if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)

# Initialize stitched_image in main scope
if 'stitched_image' not in st.session_state:
	st.session_state.stitched_image = None

# Create two columns in the main frame
col1, col2 = st.columns(2)

# In the first column, add a button for stitching images
if col1.button('Stitch Images'):
    if len(images) >= 2:
        # Initialize the stitcher
        stitcher = cv2.Stitcher.create()

        # Stitch the images
        status, stitched = stitcher.stitch(images)

        # If the status is '0', the stitching was successful
        if status == 0:
            # Convert stitched image back to RGB
            st.session_state.stitched_image = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)

            # Display the stitched image
            st.image(st.session_state.stitched_image, caption='Stitched Image', use_column_width=True)
        else:
            st.write('Error during stitching, status code = ' + str(status))
    else:
        st.write('Please upload at least two images for stitching')

# In the second column, add a button for cropping the stitched image
if col2.button('Crop Stitched Image'):
    if st.session_state.stitched_image is not None:
        # Convert to grayscale
        st.session_state.stitched_image = cv2.copyMakeBorder(st.session_state.stitched_image, 10, 10, 10, 10,
            cv2.BORDER_CONSTANT, (0, 0, 0))
		# convert the stitched image to grayscale and threshold it
		# such that all pixels greater than zero are set to 255
		# (foreground) while all others remain 0 (background)
        gray = cv2.cvtColor(st.session_state.stitched_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # allocate memory for the mask which will contain the
        # rectangular bounding box of the stitched image region
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        minRect = mask.copy()
        sub = mask.copy()
        # keep looping until there are no non-zero pixels left in the
        # subtracted image
        while cv2.countNonZero(sub) > 0:
            # erode the minimum rectangular mask and then subtract
            # the thresholded image from the minimum rectangular mask
            # so we can count if there are any non-zero pixels left
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)
        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        # use the bounding box coordinates to extract the our final
        # stitched image
        st.session_state.stitched_image = st.session_state.stitched_image[y:y + h, x:x + w]
        st.image(st.session_state.stitched_image, caption='Cropped Stitched Image', use_column_width=True)

    else:
        st.write('Please stitch the images before cropping')
