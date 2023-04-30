import os
import textwrap
from pathlib import Path
from typing import List

import cv2
import numpy as np
import PIL
from PIL import Image, ImageChops, ImageDraw, ImageFont

kMinMargin = 10


def stack_images_horizontally(images: List, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_path is not None:
        new_im.save(save_path)
    return new_im


def stack_images_vertically(images: List, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new("RGBA", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    if save_path is not None:
        new_im.save(save_path)
    return new_im


def merge_images(images: List):
    if isinstance(images[0], Image.Image):
        return stack_images_horizontally(images)

    images = list(map(stack_images_horizontally, images))
    return stack_images_vertically(images)


def draw_text(
    image: PIL.Image,
    text: str,
    font_size=None,
    font_color=(0, 0, 0),
    max_seq_length=100,
):
    W, H = image.size
    S = max(W, H)

    font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
    font_size = max(int(S / 32), 20) if font_size is None else font_size
    font = ImageFont.truetype(font_path, size=font_size)

    text_wrapped = textwrap.fill(text, max_seq_length)
    w, h = font.getsize(text_wrapped)
    new_im = Image.new("RGBA", (W, H + h))
    new_im.paste(image, (0, h))
    draw = ImageDraw.Draw(new_im)
    draw.text((max((W - w) / 2, 0), 0), text_wrapped, font=font, fill=font_color)
    return new_im


def to_white(img):
    new_img = Image.new("RGBA", img.size, "WHITE")
    new_img.paste(img, (0, 0), img)
    new_img.convert("RGB")
    return new_img


def get_bbox(in_file, fuzz=17.5):
    im = Image.open(in_file)

    # bbox = im.convert("RGBa").getbbox()
    try:
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    except OSError as err:
        print(f"error {in_file}")
        raise OSError
    diff = ImageChops.difference(im, bg)
    offset = int(round(float(fuzz) / 100.0 * 255.0))
    diff = ImageChops.add(diff, diff, 2.0, -offset)
    bbox = diff.getbbox()

    bx_min = max(bbox[0] - kMinMargin, 0)
    by_min = max(bbox[1] - kMinMargin, 0)
    bx_max = min(bbox[2] + kMinMargin, im.size[0])
    by_max = min(bbox[3] + kMinMargin, im.size[1])
    bbox_margin = (bx_min, by_min, bx_max, by_max)
    return bbox_margin


def get_largest_bbox(in_files):
    largest_bbox = (float("Inf"), float("Inf"), -float("Inf"), -float("Inf"))
    for in_file in in_files:
        bbox = get_bbox(in_file)
        largest_bbox = (
            min(bbox[0], largest_bbox[0]),
            min(bbox[1], largest_bbox[1]),
            max(bbox[2], largest_bbox[2]),
            max(bbox[3], largest_bbox[3]),
        )
    return largest_bbox


def trim(in_file, out_file, keep_ratio):
    # im = Image.open(in_file)
    # bbox = im.convert("RGBa").getbbox()
    bbox = get_bbox(in_file)
    trim_with_bbox(in_file, out_file, bbox, keep_ratio)


def trim_with_bbox(in_file, out_file, bbox, keep_ratio):
    im = Image.open(in_file)

    if keep_ratio:
        w, h = im.size
        r = float(w) / h

        bx_min, by_min, bx_max, by_max = bbox[0], bbox[1], bbox[2], bbox[3]
        bw, bh = bx_max - bx_min, by_max - by_min
        bcx, bcy = 0.5 * (bx_min + bx_max), 0.5 * (by_min + by_max)
        br = float(bw) / bh

        if br > r:
            bh = int(round(bw / r))
            by_min, by_max = int(round(bcy - 0.5 * bh)), int(round(bcy + 0.5 * bh))
            if by_min < 0:
                by_min = 0
                by_max = bh
            elif by_max > h:
                by_max = h
                by_min = h - bh
            assert bh >= bh
        elif br < r:
            bw = int(round(bh * r))
            bx_min, bx_max = int(round(bcx - 0.5 * bw)), int(round(bcx + 0.5 * bw))
            if bx_min < 0:
                bx_min = 0
                bx_max = bw
            elif bx_max > w:
                bx_max = w
                bx_min = w - bw

        bbox = (bx_min, by_min, bx_max, by_max)

    im.crop(bbox).save(out_file, "png")


def trim_with_largest_bbox(in_files, out_files, keep_ratio):
    assert len(in_files) == len(out_files)

    bbox = get_largest_bbox(in_files)
    for i in range(len(in_files)):
        trim_with_bbox(in_files[i], out_files[i], bbox, keep_ratio)


def create_image_table_tight_centering(
    in_img_files, out_img_file, max_total_width=2560, draw_col_lines=[]
):

    n_rows = len(in_img_files)
    n_cols = len(in_img_files[0])

    # Compute width and height of each image.
    width = 0
    row_top = [float("Inf")] * n_rows
    row_bottom = [-float("Inf")] * n_rows

    for row in range(n_rows):
        for col in range(n_cols):
            img_left, img_top, img_right, img_bottom = get_bbox(in_img_files[row][col])
            img_width = img_right - img_left
            width = max(width, img_width)
            row_top[row] = min(row_top[row], img_top)
            row_bottom[row] = max(row_bottom[row], img_bottom)

    row_height = [bottom - top for bottom, top in zip(row_bottom, row_top)]

    # Combine images.
    cmd = "convert "
    for row in range(n_rows):
        cmd += " \( "
        for col in range(n_cols):
            img_left, img_top, img_right, img_bottom = get_bbox(in_img_files[row][col])
            img_h_center = 0.5 * (img_left + img_right)
            left = int(img_h_center - 0.5 * width)
            cmd += " \( {} ".format(in_img_files[row][col])
            cmd += "-gravity NorthWest -crop {}x{}+{}+{} +repage \) ".format(
                width, row_height[row], left, row_top[row]
            )
        cmd += " -gravity center -background white +append \) "

    cmd += "-append " + out_img_file
    print(cmd)
    os.system(cmd)

    # Draw lines for columns.
    for col in draw_col_lines:
        if col <= 0 or col >= n_cols:
            continue
        strokewidth = max(int(round(width * 0.005)), 1)
        pos = col * width
        cmd = "convert " + out_img_file + " -stroke black "
        cmd += "-strokewidth {} ".format(strokewidth)
        cmd += '-draw "line {0},0 {0},10000000" '.format(pos) + out_img_file
        os.system(cmd)

    # Resize the combined image if it is too large.
    print(n_cols * width)
    if (n_cols * width) > max_total_width:
        cmd = "convert {0} -resize {1}x +repage {0}".format(
            out_img_file, max_total_width
        )
        print(cmd)
        os.system(cmd)

    print("Saved '{}'.".format(out_img_file))

    return width, row_height


def create_image_table_tight_centering_per_row(
    in_img_files, out_img_dir, max_total_width=1280, draw_col_lines=[]
):

    n_rows = len(in_img_files)
    n_cols = len(in_img_files[0])

    # Compute width and height of each image.
    width = 0
    row_top = [float("Inf")] * n_rows
    row_bottom = [-float("Inf")] * n_rows

    for row in range(n_rows):
        for col in range(n_cols):
            img_left, img_top, img_right, img_bottom = get_bbox(in_img_files[row][col])
            img_width = img_right - img_left
            width = max(width, img_width)
            row_top[row] = min(row_top[row], img_top)
            row_bottom[row] = max(row_bottom[row], img_bottom)

    row_height = [bottom - top for bottom, top in zip(row_bottom, row_top)]

    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    # Combine images.
    for row in range(n_rows):
        out_img_file = os.path.join(out_img_dir, "{:02d}.png".format(row))
        cmd = "convert "
        for col in range(n_cols):
            img_left, img_top, img_right, img_bottom = get_bbox(in_img_files[row][col])
            img_h_center = 0.5 * (img_left + img_right)
            left = int(img_h_center - 0.5 * width)
            cmd += " \( {} ".format(in_img_files[row][col])
            cmd += "-gravity NorthWest -crop {}x{}+{}+{} +repage \) ".format(
                width, row_height[row], left, row_top[row]
            )
        cmd += " -gravity center -background white +append " + out_img_file
        print(cmd)
        os.system(cmd)

        # Draw lines for columns.
        for col in draw_col_lines:
            if col <= 0 or col >= n_cols:
                continue
            strokewidth = max(int(round(width * 0.005)), 1)
            pos = col * width
            cmd = "convert " + out_img_file + " -stroke black "
            cmd += "-strokewidth {} ".format(strokewidth)
            cmd += '-draw "line {0},0 {0},10000000" '.format(pos) + out_img_file
            os.system(cmd)
            print(cmd)

        # Resize the combined image if it is too large.
        print(n_cols * width)
        if (n_cols * width) > max_total_width:
            cmd = "convert {0} -resize {1}x +repage {0}".format(
                out_img_file, max_total_width
            )
            print(cmd)
            os.system(cmd)

        print("Saved '{}'.".format(out_img_file))

    return width, row_height


def create_image_table_tight_centering_per_col(
    in_img_files, out_img_dir, max_width=2560, draw_col_lines=[]
):

    n_rows = len(in_img_files)
    n_cols = len(in_img_files[0])

    # Compute width and height of each image.
    width = 0
    row_top = [float("Inf")] * n_rows
    row_bottom = [-float("Inf")] * n_rows

    for row in range(n_rows):
        for col in range(n_cols):
            img_left, img_top, img_right, img_bottom = get_bbox(in_img_files[row][col])
            img_width = img_right - img_left
            width = max(width, img_width)
            row_top[row] = min(row_top[row], img_top)
            row_bottom[row] = max(row_bottom[row], img_bottom)

    row_height = [bottom - top for bottom, top in zip(row_bottom, row_top)]

    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    # Combine images.
    for col in range(n_cols):
        out_img_file = os.path.join(out_img_dir, "{:02d}.png".format(col))
        cmd = "convert "
        for row in range(n_rows):
            img_left, img_top, img_right, img_bottom = get_bbox(in_img_files[row][col])
            img_h_center = 0.5 * (img_left + img_right)
            left = int(img_h_center - 0.5 * width)
            cmd += " \( {} ".format(in_img_files[row][col])
            cmd += "-gravity NorthWest -crop {}x{}+{}+{} +repage \) ".format(
                width, row_height[row], left, row_top[row]
            )
        cmd += " -gravity center -background white -append " + out_img_file
        print(cmd)
        os.system(cmd)

        # Resize the combined image if it is too large.
        if width > max_width:
            cmd = "convert {0} -resize {1}x +repage {0}".format(out_img_file, max_width)
            print(cmd)
            os.system(cmd)

        print("Saved '{}'.".format(out_img_file))

    return width, row_height


def create_image_table_after_crop(
    in_img_files,
    out_img_file,
    lbox=None,
    tbox=None,
    rbox=None,
    dbox=None,
    max_total_width=2560,
    draw_col_lines=[],
    transpose=False,
    verbose=False,
    line_multi=None,
):
    out_img_file = str(out_img_file)
    if not isinstance(in_img_files[0], list):
        in_img_files = [in_img_files]
    in_img_files = [[x for x in row if len(str(x)) != 0] for row in in_img_files]
    if transpose:
        x = np.array(in_img_files)
        in_img_files = x.transpose().tolist()

    n_rows = len(in_img_files)
    n_cols = len(in_img_files[0])

    # Compute width and height of each image.
    width = 0
    row_top = [float("Inf")] * n_rows
    row_bottom = [-float("Inf")] * n_rows

    for row in range(n_rows):
        for col in range(n_cols):
            img_left, img_top, img_right, img_bottom = get_bbox(in_img_files[row][col])
            # img_left, img_top, img_right, img_bottom = lbox, tbox, rbox, dbox
            img_left = img_left if lbox is None else lbox
            img_top = img_top if tbox is None else tbox
            img_right = img_right if rbox is None else rbox
            img_bottom = img_bottom if dbox is None else dbox
            img_width = img_right - img_left
            width = max(width, img_width)
            row_top[row] = min(row_top[row], img_top)
            row_bottom[row] = max(row_bottom[row], img_bottom)

    row_height = [bottom - top for bottom, top in zip(row_bottom, row_top)]

    # Combine images.
    cmd = "convert "
    for row in range(n_rows):
        cmd += " \( "
        for col in range(n_cols):
            # img_left, img_top, img_right, img_bottom = lbox, tbox, rbox, dbox
            img_left, img_top, img_right, img_bottom = get_bbox(in_img_files[row][col])
            img_left = img_left if lbox is None else lbox
            img_top = img_top if tbox is None else tbox
            img_right = img_right if rbox is None else rbox
            img_bottom = img_bottom if dbox is None else dbox
            img_h_center = 0.5 * (img_left + img_right)
            left = int(img_h_center - 0.5 * width)
            cmd += " \( {} ".format(in_img_files[row][col])
            cmd += "-gravity NorthWest -crop {}x{}+{}+{} +repage \) ".format(
                width, row_height[row], left, row_top[row]
            )
        cmd += " -gravity center -background white +append \) "

    cmd += "-append " + out_img_file
    if verbose:
        print(cmd)
    os.system(cmd)
    # Draw lines for columns.
    for col in draw_col_lines:
        if col <= 0 or col >= n_cols:
            continue
        strokewidth = max(int(round(width * 0.005)), 1)
        if line_multi is not None:
            strokewidth *= line_multi
        pos = col * width
        cmd = "convert " + out_img_file + " -stroke black "
        cmd += "-strokewidth {} ".format(strokewidth)
        cmd += '-draw "line {0},0 {0},10000000" '.format(pos) + out_img_file
        if verbose:
            print(cmd)
        os.system(cmd)

    # Resize the combined image if it is too large.
    # print(n_cols * width)
    # if (n_cols * width) > max_total_width:
    # cmd = "convert {0} -resize {1}x +repage {0}".format(
    # out_img_file, max_total_width
    # )
    # print(cmd)
    # os.system(cmd)

    print("Saved '{}'.".format(out_img_file))

    return width, row_height


def make_2dgrid(input_list, num_rows=None, num_cols=None):
    # if num_rows * num_cols != len(input_list):
    # raise Warning("Number of rows and columns do not match the length of the input list.")

    if num_rows is None and num_cols is not None:
        num_rows = len(input_list) // num_cols + 1
    output_list = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            if i * num_cols + j >= len(input_list):
                break
            row.append(input_list[i * num_cols + j])
        output_list.append(row)

    return output_list
