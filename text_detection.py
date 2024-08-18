import cv2
import pytesseract
import math

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# settings
PADDING = 5
MIN_AREA = 1000 # dependent on image size?
MAX_ROWS = 4
SCALE = 2
TESTING = True

def get_dist(ax, ay, bx, by):
    return math.sqrt((ax-bx)**2 + (ay-by)**2)

def read_text(start_pos, finish_pos):
    image = cv2.imread("capture.jpg")
    image_height, image_width, _ = image.shape
    image = cv2.resize(image, (round(image_width * SCALE), round(image_height * SCALE)))
    image_height, image_width, _ = image.shape

    print(image_height, image_width)

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

    max_x = 0
    max_y = 0
    min_x = 100000
    min_y = 100000

    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        x, y, w, h = (x-PADDING, y-PADDING, w+2*PADDING, h+2*PADDING)
        if x + w > image_width:
            w = image_width - x
        if y + h > image_height:
            h = image_height - y
        if x < 0: x = 0
        if y < 0: y = 0
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
            output.append(line_text.strip())
            index += 1
            if line_text.strip() != '':
                min_x = min(min_x, x)
                max_x = max(max_x, x+w)
                min_y = min(min_y, y)
                max_y = max(max_y, y+h)
                # if max_x == x+w:
                #     print(line_text.strip(), index)
        # output.append("\n")

    print(output)

    print(median_height, "mde")

    if TESTING:
        # draw bounding boxes
        index = 0
        for row in boxes_rows[:MAX_ROWS+1]:
            for x, y, w, h in row:
                colour = (0, 0, index * 50) if index % 2 else (index * 50, 0, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
            index += 1

    # cv2.rectangle(image, (round(start_pos.x * image_width), round(start_pos.y * image_height)), (round(finish_pos.x * image_width), round(finish_pos.y * image_height)), (0, 255, 0), 2)
    # sx, sy, fx, fy = (0.442717969, 0.453878403, 0.907342076, 0.885900795)
    sx, sy = start_pos.x, start_pos.y
    fx, fy = finish_pos.x, finish_pos.y
    sx = round(sx * image_width)
    sy = round(sy * image_height) - median_height
    fx = round(fx * image_width)
    fy = round(fy * image_height) - median_height

    best_dist_s = 1000000
    best_dist_f = 1000000
    nsx, nsy = -1, -1
    nfx, nfy = -1, -1
    start_idx = -1
    finish_idx = -1
    for idx, box in enumerate(boxes):
        if get_dist(box[0], box[1], sx, sy) < best_dist_s:
            best_dist_s = get_dist(box[0], box[1], sx, sy)
            nsx, nsy = box[0], box[1]
            start_idx = idx
        if get_dist(box[0]+box[2], box[1]+box[3], fx, fy) < best_dist_f:
            best_dist_f = get_dist(box[0]+box[2], box[1]+box[3], fx, fy)
            nfx, nfy = box[0]+box[2], box[1]+box[3]
            finish_idx = idx

    sx, sy = nsx, nsy
    fx, fy = nfx, nfy

    text_output = ' '.join(output[start_idx:finish_idx+1])
    f = open("notes.txt", "a")
    f.write(text_output + "\n\n")
    f.close()

    # cv2.rectangle(image, (round(sx * image_width), round(sy * image_height) - median_height), (round(fx * image_width), round(fy * image_height) - median_height), (0, 255, 0), -1)
    cv2.line(image, (sx, sy), (sx + median_height, sy), (0, 255, 0), 2)
    cv2.line(image, (sx, sy), (sx, sy + median_height), (0, 255, 0), 2)
    cv2.line(image, (fx, fy), (fx, fy - median_height), (0, 255, 0), 2)
    cv2.line(image, (fx, fy), (fx - median_height, fy), (0, 255, 0), 2)
    # cv2.line(image, (sx, sy), (fx, fy), (0, 255, 0), 2)
    # cv2.line(image, (sx, sy), (max_x, sy), (0, 255, 0), 2)
    # cv2.line(image, (max_x, sy), (max_x, fy), (0, 255, 0), 2)
    # cv2.line(image, (max_x, fy), (fx, fy), (0, 255, 0), 2)
    # cv2.line(image, (fx, fy), (fx, fy + median_height), (0, 255, 255), 2)
    # cv2.line(image, (fx, fy + median_height), (min_x, fy + median_height), (0, 255, 0), 2)
    # cv2.line(image, (min_x, fy + median_height), (min_x, sy + median_height), (0, 255, 0), 2)
    # cv2.line(image, (min_x, sy + median_height), (sx, sy + median_height), (0, 255, 0), 2)
    # cv2.line(image, (sx, sy + median_height), (sx, sy), (0, 255, 0), 2)
    # print((round(sx * image_width), round(sy * image_height)), (round(fx * image_width), round(fy * image_height)))

    # cv2.drawContours(image, filtered_contours, -1, (0, 255, 0))
    cv2.imshow('Highlighted passage', image)
    cv2.waitKey(0)
    cv2.imwrite('result.jpg',image)

# read_text({"x": 0.442717969, "y": 0.453878403}, {"x": 0.907342076, "y": 0.885900795})