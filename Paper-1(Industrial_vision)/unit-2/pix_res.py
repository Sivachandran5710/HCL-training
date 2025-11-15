import math

# Known parameters
sensor_width_mm = 6.3         # mm (typical smartphone sensor width)
aperture_N = 2.8              # f-number
circle_of_confusion = 0.005   # mm
subject_distance_mm = 1000    # 1 meter

#  Known object in image 
real_object_width_mm = 100.0  # known width (e.g., 10 cm ruler)
pixel_x1 = 200                # starting pixel
pixel_x2 = 600                # ending pixel
pixel_width = pixel_x2 - pixel_x1

focal_length_mm = 8.0
pixel_resolution = real_object_width_mm / pixel_width
print(f"Pixel Resolution: {pixel_resolution:.4f} mm/pixel")

# Depth of Field (DOF)
dof = (2 * aperture_N * circle_of_confusion * subject_distance_mm**2) / (focal_length_mm**2)
print(f"Estimated DOF: {dof:.2f} mm")
