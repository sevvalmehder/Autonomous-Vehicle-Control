import argparse
import collections
import datetime
import glob
import time
import copy
from collections import deque
import logging
import math
import os
import random
import re
import sys
import weakref
import matplotlib.pyplot as plt
import carla 
import torch

from agent import HazineAgent


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)'
    )
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)'
    )
    argparser.add_argument(
        '-m', '--model',
        default="./hazine_model/model_last.h5",
        type=str
    )

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    
    # Create an agent
    agent = HazineAgent(args.model)
    print("Hazine Agent created")

    try:
        sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass

    from interface import hazine_interface

    print("The game loop start")
    hazine_interface.game_loop(args, agent)










