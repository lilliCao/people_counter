#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.input_blob = None
        self.exec_network = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files
        Default device: CPU
        '''
        # Get .xml and .bin file
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initalize the plugin
        self.plugin = IECore()

        # Add an extension if given
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read IR into IENetwork
        network = IENetwork(model=model_xml, weights=model_bin)

        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so
        supported_layers = self.plugin.query_network(network=network, device_name=device)
        unsupported_layers = [l for l in network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        # Load IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network, device)

        # Get the input layer
        self.input_blob = next(iter(network.inputs))

        # Return the input shape for preprocessing
        return network.inputs[self.input_blob].shape

    def sync_inference(self, image):
        '''
        Make a synchronous inference request
        '''
        self.exec_network.infer({self.input_blob: image})
        return

    def async_inference(self, image, idx=0):
        '''
        Make a asynchronous inference request
        '''
        self.exec_network.start_async(request_id=idx, inputs={self.input_blob: image})
        return

    def wait(self, idx=0):
        '''
        Wait for an asynchronous request to be finished
        '''
        return self.exec_network.requests[idx].wait(-1)

    def get_output(self, idx=0):
        '''
        Get output from the IE
        '''
        return self.exec_network.requests[idx].outputs
