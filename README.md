# cycle-gan-keras

A simple keras implementation of CycleGAN (https://arxiv.org/pdf/1703.10593.pdf) for unpaired image translation.

This implementation favors minor changes due to the generous application of skip connections throughout the generator network. The discriminator is also trained to distinguish real images of the two classes to avoid it learning to simply detect visual artifacts. 

Directions:
* Load images of one class into images_a and images of the second class into images_b
* Run the training routine to build the networks
* Run the testing routine to output translated versions of all images with discriminator confidences

A few tips:
* Narrower networks train more smoothly and are less prone to mode collapse; start at 8 nodes per layer
* Loss is not a great indicator of performance in adverserial setups; visualize the outputs periodically
* Keep the batch size for training very low or it will likely fail to learn

Claude Monet <-> William Turner:
