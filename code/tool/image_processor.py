import os
import requests
import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self, image_path=''):
        if(image_path != ''):
            self.image_path = image_path
            self.image = Image.open(image_path)
            self.image_np = np.array(self.image)
            self.width, self.height = self.image_np.shape[1], self.image_np.shape[0]


    @staticmethod
    def check_column_white(column_pixels):
        """Check if the column is almost white."""
        is_almost_white = np.logical_or(column_pixels == 254, column_pixels == 255)
        white_pixels_ratio = np.mean(np.all(is_almost_white, axis=-1))
        return white_pixels_ratio >= 0.98  # At least 98% of the column's pixels are white


    def find_white_section(self, start, end):
        """Find white sections in a specific range."""
        white_sections = []
        in_white_section = False
        start_index = 0

        for col in range(start, end):  # Only search within the specified range
            column_pixels = self.image_np[:, col, :]
            if self.check_column_white(column_pixels):
                if not in_white_section:
                    start_index = col
                    in_white_section = True
            else:
                if in_white_section:
                    white_sections.append((start_index, col))
                    in_white_section = False

        if in_white_section:
            white_sections.append((start_index, end))

        return white_sections


    def split_image(self):
        start_col = self.width * 2 // 5
        end_col = self.width * 3 // 5
        white_sections = self.find_white_section(start_col, end_col)

        if white_sections:
            middle_section = white_sections[len(white_sections) // 2]
            mid_col = (middle_section[0] + middle_section[1]) // 2
        else:
            raise ValueError("No suitable white column found within the specified range")

        left_box = (0, 0, mid_col, self.height)
        right_box = (mid_col, 0, self.width, self.height)
        left_image = self.image.crop(left_box)
        right_image = self.image.crop(right_box)

        save_dir, filename = os.path.split(self.image_path)
        base, extension = os.path.splitext(filename)

        left_image_path = os.path.join(save_dir, base + '_front' + extension)
        right_image_path = os.path.join(save_dir, base + '_back' + extension)
        left_image.save(left_image_path)
        right_image.save(right_image_path)

        return left_image_path, right_image_path
    

    def stitch_images(self, image_paths, output_path):
        if not image_paths:
            raise ValueError("No image paths provided")
        sample_image = Image.open(image_paths[0])
        single_width, single_height = sample_image.size
        num_images = len(image_paths)
        total_desired_width = single_width  # The final width is equal to the width of a single image
        total_current_width = single_width * num_images
        total_width_to_cut = max(0, total_current_width - total_desired_width)
        width_to_cut_per_image = total_width_to_cut // num_images
        stitched_image = Image.new('RGB', (total_desired_width, single_height), "white")
        current_x = 0
        
        for path in image_paths:
            image = Image.open(path)
            # If cropping is needed, crop each image according to the calculated result
            if width_to_cut_per_image > 0:
                left_margin = width_to_cut_per_image // 2
                right_margin = image.width - width_to_cut_per_image + left_margin
                image = image.crop((left_margin, 0, right_margin, image.height))
            stitched_image.paste(image, (current_x, 0))
            current_x += image.width
        
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        stitched_image.save(output_path)
        return output_path
    

    def download_image(self, image_url, save_path):
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)


    def resize_image(self, image_path):
        original_image = Image.open(image_path)
        width, height = original_image.size
        top_blank_height = height // 2
        final_height = height + top_blank_height
        final_width = int(final_height * 5 / 3)
        new_image = Image.new("RGB", (final_width, final_height), color="white")
        left = (final_width - width) // 2
        top = top_blank_height
        new_image.paste(original_image, (left, top))
        new_image.save(image_path)
        return image_path


    def has_black_borders(self, image_path, threshold=10, black_limit=20):
        img = Image.open(image_path)
        pixels = img.load()
        width, height = img.size
        def is_black_pixel(pixel):
            return all(x <= black_limit for x in pixel)
        # Check the top and bottom borders
        for y in range(threshold):
            if all(is_black_pixel(pixels[x, y]) for x in range(width)):
                return True
            if all(is_black_pixel(pixels[x, height - 1 - y]) for x in range(width)):
                return True
        # Check the left and right borders
        for x in range(threshold):
            if all(is_black_pixel(pixels[x, y]) for y in range(height)):
                return True
            if all(is_black_pixel(pixels[width - 1 - x, y]) for y in range(height)):
                return True
        return False
