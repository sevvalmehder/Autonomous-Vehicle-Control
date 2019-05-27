import numpy as np
import scipy
import sys
import os
import glob

from scipy.misc import imresize
from PIL import Image

import matplotlib.pyplot as plt

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import math
from hazine_model.model import pilotNet

class HazineAgent(object):

    def __init__(self, model_file):
        
        # Load the model architecture and model weights
        self.model = pilotNet()
        self.model.load_weights(model_file)
        
        # Create a latest image variable
        self.latest_image = None

        
    def is_within_distance_ahead(self, target_location, current_location, orientation, max_distance):
        """
        Check if a target object is within a certain distance in front of a reference object.

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :return: True if target object is within max_distance ahead of the reference object
        """
        target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
        norm_target = np.linalg.norm(target_vector)
        if norm_target > max_distance:
            return False

        forward_vector = np.array(
            [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
        d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

        return d_angle < 90.0

    def _is_light_red_europe_style(self, world, lights_list):
        """
        This method is specialized to check European style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                  affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """

        proximity_threshold = 10.0 # meters

        ego_vehicle_location = world.vehicle.get_location()
        ego_vehicle_waypoint = world.world.get_map().get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_waypoint = world.world.get_map().get_waypoint(traffic_light.get_location())
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                            object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            loc = traffic_light.get_location()
            if self.is_within_distance_ahead(loc, ego_vehicle_location,
                                        world.vehicle.get_transform().rotation.yaw,
                                        proximity_threshold):
                if traffic_light.state == carla.libcarla.TrafficLightState.Red:
                    return (True, traffic_light)

        return (False, None)

    def _is_light_red(self, world, lights_list):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        _map = world.world.get_map()
        if _map.name == 'Town01' or _map.name == 'Town02':
            return self._is_light_red_europe_style(world, lights_list)
        '''
        else:
            return self._is_light_red_us_style(lights_list)
        '''

    def run_step(self, world, sensor_data, target):
        """
            Run a step on the benchmark simulation
        Args:
            world
            sensor_data: RGB camera data
            directions: The directions, high level commands

        Returns:
            Controls for the vehicle on the CARLA simulator.

        """

        # Is there a stop factor?
        stop_detected = False

        # Assign the current sensor data as a latest image
        self.latest_image = sensor_data

        '''
        # Traffic light control
        actor_list = world.world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        
        
        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(world, lights_list)
        print(light_state)
        
        if light_state:
            print("RED LIGHT ALARM")
            stop_detected = True
        '''

        if not stop_detected:

            # Create control object
            control = carla.VehicleControl()

            # Resize the latest image
            image = scipy.misc.imresize(self.latest_image, size=(66, 200))
            #scipy.misc.imsave("image.png", image)
            image = np.array([image])

            # Predict the steering angle class
            pred = self.model.predict(image)
            steer_class = np.argmax(pred[0])
            print("prediction: {}".format(steer_class))

            # Convert this predicted class to control value
            # If the predicted clas is 9, its correspond control value is 0.9
            if steer_class <= 9:
                steer = steer_class/10
            else:
                steer_class -= 9
                steer = steer_class/10
                steer *= -1

            throttle = 0.5
            brake = 0

            print("steerclass: {}, steer: {}".format(steer_class, steer))
            #steer, throttle, brake = self._process_model_outputs([steer, throttle, brake])

            # Assign control values
            control.steer = float(steer)
            control.throttle = float(throttle)
            control.brake = float(brake)


        else:
            # Assign control values for stop
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False

        return control

    def get_attentions(self, layers=None):
        """

        Returns
            The activations obtained from the first layers of the latest iteration.

        """
        if layers is None:
            layers = [0, 1, 2]
        '''
        if self.latest_image_tensor is None:
            raise ValueError('No step was ran yet. '
                             'No image to compute the activations, Try Running ')
        '''
        all_layers = self._model.get_perception_layers(self.latest_image_tensor)
        cmap = plt.get_cmap('inferno')
        attentions = []
        for layer in layers:
            y = all_layers[layer]
            att = torch.abs(y).mean(1)[0].data.cpu().numpy()
            att = att / att.max()
            att = cmap(att)
            att = np.delete(att, 3, 2)
            attentions.append(imresize(att, [88, 200]))
        return attentions

    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        if brake < 0.05:
            brake = 0.0

        if throttle > brake:
            brake = 0.0


        return steer, throttle, brake


    def _process_model_outputs_wp(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        wpa1, wpa2, throttle, brake = outputs[3], outputs[4], outputs[1], outputs[2]
        if brake < 0.2:
            brake = 0.0

        if throttle > brake:
            brake = 0.0

        steer = 0.7 * wpa2

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        return steer, throttle, brake

    def _get_oracle_prediction(self, measurements, target):
        # For the oracle, the current version of sensor data is not really relevant.
        control, _, _, _, _ = self.control_agent.run_step(measurements, [], [], target)

        return control.steer, control.throttle, control.brake


        