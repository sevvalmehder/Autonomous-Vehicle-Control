import sys 
import glob 
import os

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy 
import argparse
import pygame
import numpy as np
import cv2

try:
    import queue
except ImportError:
    import Queue as queue

from helpers import image_convert as ic
from helpers import pygame_draw as pd


def main():

	# Prepare output folder
	import time
	if args.save_images_to_disk:
		output_dir = os.path.join("output", str(time.time()))
		os.makedirs(output_dir)
	
	# Initialize the pygame
	pygame.init()

	# Connect with client
	client = carla.Client('localhost', 2000)
	client.set_timeout(10.0)

	# Create a list for keep the sensors
	sensorlist = []

	try:
		# Get a world and set the start position
		world = client.get_world()
		m = world.get_map()
		start_pose = random.choice(m.get_spawn_points())
		waypoint = m.get_waypoint(start_pose.location)

		# Get a blueprint library
		blueprint_library = world.get_blueprint_library()


		# Add vehicle
		vehicle_bp = random.choice(blueprint_library.filter('vehicle.ford.mustang'))
		vehicle = world.spawn_actor(vehicle_bp, start_pose)
		vehicle.set_simulate_physics(False)
		

		#
		# Add sensors
		#

		# Set location 
		location = carla.Transform(carla.Location(x=0, z=2.8), carla.Rotation(pitch=-15))

		# Add RGB Camera
		camera_bp = blueprint_library.find('sensor.camera.rgb')
		camera_bp.set_attribute('image_size_x', '800')
		camera_bp.set_attribute('image_size_y', '600')
		
		camera_front = world.spawn_actor(camera_bp, location, attach_to=vehicle)
		sensorlist.append(camera_front)

		'''
		# Add sementic segmentation camera
		camera_seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
		camera_seg_bp.set_attribute('image_size_x', '400')
		camera_seg_bp.set_attribute('image_size_y', '300')

		camera_seg = world.spawn_actor(camera_seg_bp,location, attach_to=vehicle)
		sensorlist.append(camera_seg)

		# Add depth camera
		camera_depth_bp = blueprint_library.find('sensor.camera.depth')
		camera_depth_bp.set_attribute('image_size_x', '400')
		camera_depth_bp.set_attribute('image_size_y', '300')

		camera_depth = world.spawn_actor(camera_depth_bp, location, attach_to= vehicle)
		sensorlist.append(camera_depth)
		'''

		# Create queues for sensor outputs and listen 
		image_queue_rgb = queue.Queue()
		camera_front.listen(image_queue_rgb.put)

		'''
		image_queue_seg = queue.Queue()
		camera_seg.listen(image_queue_seg.put)

		image_queue_depth = queue.Queue()
		camera_depth.listen(image_queue_depth.put)
		'''
		
		# Create display
		display = pygame.display.set_mode((800, 600))
		#display = pygame.display.set_mode((400, 300))

		count = 0
		while True:
			rgb_image = image_queue_rgb.get()
			#seg_image = image_queue_seg.get()
			#depth_image = image_queue_depth.get()

			# Set the next point
			waypoint = random.choice(waypoint.next(2))
			vehicle.set_transform(waypoint.transform)

			pd.draw_image(display, rgb_image, (0,0))
			#pd.draw_image(display, seg_image, (0,300))
			#pd.draw_image(display, depth_image, (400,0))
			#converted_seg = ic.labels_to_cityscapes_palette(seg_image)
			#pd.draw_array(display, converted_seg, (400,300))

			pygame.display.flip()
			#time.sleep(0.09)

			# Save the images to disk if requested.
			if args.save_images_to_disk:
				print("images saving..")
				#converted_rgb = ic.to_rgb_array(rgb_image)
				#cv2.imwrite('output/rgb/umm_%06d.png' % count, converted_rgb)
				rgb_image.save_to_disk("{path}/rgb/{number:06}.png".format(path=output_dir, number=count))
				#seg_image.save_to_disk('output/seg/umm_%06d.png' % count)
				#depth_image.save_to_disk('output/depth/%06d.png' % depth_image.frame_number)
				#cv2.imwrite('output/con_seg/umm_%06d.png' % count, converted_seg)
			count += 1

	finally:
		# Destroy the vehicle
		vehicle.destroy()
		print("vehicle destroyed")
		# Destroy the sensors
		client.apply_batch([carla.command.DestroyActor(x.id) for x in sensorlist])


if __name__ == '__main__':

	# Control the arguments
	argparser = argparse.ArgumentParser()
	argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images to disk')
	args = argparser.parse_args()


	try:	
		# Call the main
		main()
	finally:
		print("GoodBye!")