import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# settings
PADDING = 10
MIN_AREA = 1000 # dependent on image size?
MAX_ROWS = 16

image = cv2.imread("img3.jpg")

# process image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel_size = (15, 1) 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(bw_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filtered_contours = []

for contour in contours:
    if cv2.contourArea(contour) > MIN_AREA:
        filtered_contours.append(contour)

filtered_contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1])

heights = []
boxes = []

for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    x, y, w, h = (x-PADDING, y-PADDING, w+2*PADDING, h+2*PADDING) 
    if w * h > MIN_AREA:
        heights.append(h)
        boxes.append((x, y, w, h))

median_height = sorted(heights)[len(heights)//2]

boxes_rows = [[boxes[0]]]

# sort bounding boxes into rows
index = 1

for box in boxes[1:]:
    x_diff = abs(boxes_rows[-1][-1][0] - box[0])    
    y_diff = abs(boxes_rows[-1][-1][1] - box[1])
    if y_diff < (box[3] + boxes_rows[-1][-1][3]) // 4:
        boxes_rows[-1].append(box)
    else:
        boxes_rows[-1].sort(key=lambda b: b[0])
        boxes_rows.append([box])
    index += 1


output = []

# convert each box into text
index = 0
for row in boxes_rows[:MAX_ROWS+1]:
    for x, y, w, h in row:
        line_image = image[y:y + h, x:x+w]
        line_text = pytesseract.image_to_string(line_image, config='--psm 7')
        output.append((line_text.strip(), index))
        index += 1
    output.append("\n")

print(output)

print(median_height, "mde")

# draw bounding boxes
index = 0
for row in boxes_rows[:MAX_ROWS+1]:
    for x, y, w, h in row:
        colour = (0, 0, index * 50) if index % 2 else (index * 50, 0, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
    index += 1

# cv2.drawContours(image, filtered_contours, -1, (0, 255, 0))
cv2.imshow('Text Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result.jpg',image)