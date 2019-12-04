from .grid import BaseGrid
from .particles import BaseParticles

# import different backends - each backend should register in a container
import nata.backends.osiris
