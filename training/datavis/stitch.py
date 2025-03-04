from PIL import Image
import os


def merge_images(image1, image2):

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result


if __name__ == '__main__':
    root = 'plots'

    for i in range(2, 11, 2):
        merged_files = ['cfm-{}.png'.format(i-1), 'cfm-{}.png'.format(i)]
        image = None
        for fname in merged_files:
            fpath = os.path.join(root, fname)
            cur_image = Image.open(fpath)
            if image is None:
                image = cur_image
            else:
                image = merge_images(image, cur_image)

        image.save(os.path.join(root, 'stitch-{}.png'.format(i)))