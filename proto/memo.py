import cv2
import numpy as np

FILE_PATH = "../Database/IMG_3086.jpeg"


def start(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

    debug = True

    print("filters...")
    cards1, filtered = first_detection(img, debug)
    cards2, filtered2 = second_detection(img, debug)
    cards3, filtered3 = third_detection(img, debug)
    # cards4, filtered4 = fourth_detection(img, debug)

    if debug:
        cv2.imshow("filtered", filtered)
        cv2.imshow("filtered2", filtered2)
        cv2.imshow("filtered3", filtered3)
        # cv2.imshow('filtered4', filtered4)

    print("merge...")
    all_cards = cards1 + cards2 + cards3
    merged, merged_img = merge_cards(all_cards, img, debug)

    print("final check...")
    cards_checked, checked_img = final_card_check(merged, img, debug)

    if debug:
        cv2.imshow("original", img)
        cv2.imshow("merged", merged_img)
        cv2.imshow("checked", checked_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("crop...")
    crop_image(cards_checked, img, debug)

    print("image detection done.")


def approx_card(approx, img_area):
    if len(approx) < 4 or len(approx) > 6:
        # if (len(approx) != 4):
        return False

    # Area check
    area = cv2.contourArea(approx)
    if area < 1000 or area > img_area * 0.1:
        return False

    # Ratio check
    _x, _y, w, h = cv2.boundingRect(approx)
    ratio = w / h
    return not (ratio < 0.5 or ratio > 2.0)


def first_detection(img, debug):
    filtered = cv2.GaussianBlur(img, (11, 11), 0)
    filtered = cv2.Canny(filtered, 50, 150)

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    filtered_size = filtered.shape  # 3d
    filtered_area = filtered_size[0] * filtered_size[1]

    cards = []

    for i, contour in enumerate(contours):
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if approx_card(approx, filtered_area):
            if debug and len(approx) > 4:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)

                cv2.drawContours(filtered, [box], 0, (0, 255, 0), 2)
            else:
                cards.append(approx.reshape(4, 2))

                if debug:
                    cv2.drawContours(
                        filtered, [approx.reshape(4, 2)], 0, (0, 255, 0), 2
                    )
                    for x, y in approx.reshape(4, 2):
                        cv2.circle(filtered, (x, y), 6, (255, 0, 0), -1)

    return cards, filtered


def second_detection(img, debug):
    filtered = cv2.GaussianBlur(img, (9, 9), 0)
    filtered = cv2.Canny(filtered, 20, 120)

    # Morphological closing
    k = np.ones((3, 3), np.uint8)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, k, iterations=2)
    filtered = cv2.dilate(filtered, k, iterations=1)

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    filtered_size = filtered.shape  # 3d
    filtered_area = filtered_size[0] * filtered_size[1]

    cards = []

    for i, contour in enumerate(contours):
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = contour, epsilon, True

        if approx_card(approx, filtered_area):
            if len(approx) > 4:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cards.append(box)

                if debug:
                    cv2.drawContours(filtered, [box], 0, (0, 255, 0), 2)
            else:
                cards.append(approx.reshape(4, 2))

                if debug:
                    cv2.drawContours(
                        filtered, [approx.reshape(4, 2)], 0, (0, 255, 0), 2
                    )
                    for x, y in approx.reshape(4, 2):
                        cv2.circle(filtered, (x, y), 6, (255, 0, 0), -1)

    return cards, filtered


def third_detection(img, debug):
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    filtered = cv2.Canny(gray, 40, 140)

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    filtered_size = filtered.shape  # 3d
    filtered_area = filtered_size[0] * filtered_size[1]

    cards = []

    for i, contour in enumerate(contours):
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if approx_card(approx, filtered_area):
            if len(approx) > 4:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cards.append(box)
                if debug:
                    cv2.drawContours(filtered, [box], 0, (0, 255, 0), 2)
            else:
                cards.append(approx.reshape(4, 2))

                if debug:
                    cv2.drawContours(
                        filtered, [approx.reshape(4, 2)], 0, (0, 255, 0), 2
                    )
                    for x, y in approx.reshape(4, 2):
                        cv2.circle(filtered, (x, y), 6, (255, 0, 0), -1)

    return cards, filtered


def card_center(card):
    return np.mean(card, axis=0)


def merge_cards(cards, img, debug, dist_thresh=30):
    merged = []

    for card in cards:
        c_center = card_center(card)
        found = False

        for i, m in enumerate(merged):
            m_center = card_center(m)
            if np.linalg.norm(c_center - m_center) < dist_thresh:
                found = True
                break

        if not found:
            merged.append(card)

    result = None
    if debug:
        result = img.copy()
        for card in merged:
            rect = cv2.minAreaRect(card.astype(np.float32))
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(result, [box], 0, (0, 255, 0), 2)

    return merged, result


def cards_area_mean(cards):
    areas = []
    for card in cards:
        areas.append(cv2.contourArea(card))
    return np.mean(areas)


def final_card_check(cards, img, debug):
    mean = cards_area_mean(cards)
    cards_checked = []
    for card in cards:
        if abs(cv2.contourArea(card) - mean) < 9000:
            cards_checked.append(card)

    result = None
    if debug:
        result = img.copy()
        for card in cards_checked:
            rect = cv2.minAreaRect(card.astype(np.float32))
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(result, [box], 0, (0, 255, 0), 2)

    return cards_checked, result


def crop_image(cards, img, debug):
    h, w = img.shape[:2]

    for i, card in enumerate(cards):
        x_coords = card[:, 0]
        y_coords = card[:, 1]

        x_min = max(0, np.min(x_coords) - 20)
        x_max = min(w, np.max(x_coords) + 20)
        y_min = max(0, np.min(y_coords) - 20)
        y_max = min(h, np.max(y_coords) + 20)

        card_pic = img[y_min:y_max, x_min:x_max]
        cv2.imwrite(f"card_{i}.jpg", card_pic)

        if debug:
            cv2.imshow("Cropped Image", card_pic)
            cv2.waitKey(0)
    cv2.destroyAllWindows()


def fourth_detection(img, debug):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = cv2.convertScaleAbs(magnitude)

    # Threshold to get edges
    _, filtered = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    k = np.ones((3, 3), np.uint8)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, k, iterations=2)
    filtered = cv2.dilate(filtered, k, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    filtered_size = filtered.shape
    filtered_area = filtered_size[0] * filtered_size[1]

    cards = []

    for contour in contours:
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if approx_card(approx, filtered_area):
            if len(approx) > 4:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cards.append(box)

                if debug:
                    cv2.drawContours(filtered, [box], 0, (0, 255, 0), 2)
            else:
                cards.append(approx.reshape(4, 2))

                if debug:
                    cv2.drawContours(
                        filtered, [approx.reshape(4, 2)], 0, (0, 255, 0), 2
                    )
                    for x, y in approx.reshape(4, 2):
                        cv2.circle(filtered, (x, y), 6, (255, 0, 0), -1)

    return cards, filtered


if __name__ == "__main__":
    start(FILE_PATH)
