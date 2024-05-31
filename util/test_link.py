import os
import sys
from tools import object_completion_network


def main():
    comp_network = object_completion_network()
    comp_network.complete()


if __name__ == '__main__':
    main()
