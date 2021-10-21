import sys
from itertools import count, combinations

import cv2
import matplotlib.pyplot as plt
import numpy as np

save_animation = ("animate" in sys.argv)


def inrange_thresh(image, color, thresh, binarize_thresh=None):
    """
    Apply cv.inRange with a threshold near the given color, optionally threshold the final image.
    """
    min_color = tuple(c - thresh for c in color)
    max_color = tuple(c + thresh for c in color)
    image = cv2.inRange(image, min_color, max_color)
    if binarize_thresh is not None:
        t, image = cv2.threshold(image, binarize_thresh, 255, cv2.THRESH_BINARY)
    return image


def find_feature_bboxes(image):
    """
    Find contours in the image and return their bounding boxes.
    :return: Iterable of (x, y, w, h)
    """
    cnts, *_ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        yield cv2.boundingRect(c)


def distance_from_point(input_mask, start_point):
    """
    Build a distance map following truthy paths in the input mask image, starting from start_point.
    :return: Tuple of distance map matrix and infinity value for the matrix
    """
    binary_mask = (input_mask > 127)
    # Figure out a suitably large number to serve as "infinity" for the mask.
    infinite_distance = max(binary_mask.shape) * 2

    # Generate a distance map with unreachable points, then seed it with our start point.
    dist_map = np.full_like(input_mask, infinite_distance, dtype="uint32")
    dist_map[start_point[::-1]] = 0

    # Precompute a structuring element we can use to dilate the "wavefront" to walk along the route with.
    struct_el = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for step in count(1):
        # Compute a zero map for new neighboring pixels.
        neighbor_map = np.full_like(dist_map, 0, dtype="uint8")
        # Mask in all of the pixels that were filled in by the last step.
        neighbor_map[dist_map == (step - 1)] = 255
        # Dilate the map with the structuring element so new neighboring pixels would be included.
        neighbor_map = cv2.dilate(neighbor_map, struct_el)

        # Figure out which pixels in the dist map should be filled
        new_dist_mask = (
            (dist_map > step) &  # must be more distant than we've already filled
            (neighbor_map > 0) &  # must be one of these new neighbors
            binary_mask  # must be walkable
        )
        if not np.any(new_dist_mask):
            # If there are no new pixels, we're done.
            break
        dist_map[new_dist_mask] = step

        if save_animation:
            plt.imsave(f"anim/nm-{start_point[0]}-{start_point[1]}-{step:05d}.png", neighbor_map)
            plt.imsave(f"anim/dm-{start_point[0]}-{start_point[1]}-{step:05d}.png", dist_map)
    return (dist_map, infinite_distance)


def print_distance_matrix(distances):
    station_ids = sorted({k[0] for k in distances})
    print("B / A  | " + " ".join(f"#{x}".rjust(5) for x in station_ids))
    for sb in station_ids:
        formatted_ds = [str(distances.get(tuple(sorted((sa, sb))), "-")).rjust(5) for sa in station_ids]
        print(f"#{sb}".rjust(6) + " | ", *formatted_ds)


def main():
    image = cv2.imread("hHwyu.png", cv2.IMREAD_COLOR)

    marker_color = (239, 30, 40)[::-1]  # RGB -> BGR
    route_color = (0, 0, 0)[::-1]

    # Grab a grayscale image of the markers only
    markers = (inrange_thresh(image, marker_color, 5, 5) > 0).astype(np.uint8)
    # Use the center of their bounding boxes as a station location
    station_positions = [(int(x + w / 2), int(y + h / 2)) for x, y, w, h in find_feature_bboxes(markers)]
    station_positions.sort(key=lambda pair: pair[1])

    # Dilate the markers a bit so they'll always sit on the roads, then splat them on.
    # We'll use this as the base map for the contour-walking algorithm so it's all connected.
    markers_dilated = cv2.dilate(markers, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    routes_binary = inrange_thresh(image, route_color, 25, 0)
    routes_binary[markers_dilated > 0] = 255

    plt.imsave("out/routes_binary.png", routes_binary, dpi=300)

    station_infos = []
    for station_id, start_point in enumerate(station_positions, 1):
        print(f"Computing distance map for station {station_id} at {start_point}")
        distmap, inf = distance_from_point(routes_binary, start_point)
        station_infos.append((station_id, start_point, distmap, inf))

    distances = {}
    for (sa_id, sa_point, sa_map, sa_inf), (sb_id, sb_point, sb_map, sb_inf) in combinations(station_infos, 2):
        distance = sa_map[sb_point[::-1]]
        if distance >= sa_inf:
            distance = np.inf
        distances[tuple(sorted((sa_id, sb_id)))] = distance
        print(f"Distance between {sa_id} ({sa_point}) and {sb_id} ({sb_point}): {distance}")

    print_distance_matrix(distances)

    fig, axs_grid = plt.subplots(4, len(station_positions) // 4, figsize=(15, 10))
    for subplot, (station_id, start_point, distmap, inf) in zip(axs_grid.flat, station_infos):
        # Prettify the distmap for display (invert, colorize)
        colorized_distmap = plt.cm.viridis(plt.Normalize()(inf - distmap))
        # Draw the start point marker circle
        cv2.circle(colorized_distmap, start_point, 15, (255, 255, 255))
        subplot.set_title(f"#{station_id} {start_point}")
        subplot.imshow(colorized_distmap)
        subplot.axis("off")

    plt.tight_layout(pad=.7)
    # plt.show()
    plt.savefig("out/all_distmaps.png", dpi=500)


if __name__ == '__main__':
    main()
