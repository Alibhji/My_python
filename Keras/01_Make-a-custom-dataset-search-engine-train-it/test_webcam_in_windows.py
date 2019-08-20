import numpy as np
import cv2
from PIL import Image
from io import BytesIO

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    img = Image.fromarray(frame)
    new_width  = 1280
    new_height = 720
    img = img.resize((new_width, new_height), Image.BICUBIC)
    frame = np.array(img)
    model_image_size = (608, 608)
    resized_image = img.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    b = BytesIO()
    img.save(b,format="jpeg")
    image=Image.open(b)
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data,
                                                                                input_image_shape: [image.size[1], image.size[0]],
                                                                                K.learning_phase(): 0})
    
    
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()