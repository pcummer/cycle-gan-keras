# cycle-gan-keras

A simple keras implementation of CycleGAN (https://arxiv.org/pdf/1703.10593.pdf)

This implementation favors minor changes due to the generous application of skip connections throughout the generator network. The discriminator is also trained to distinguish real images of the two classes to avoid it learning to simply detect visual artifacts. 
