# Music Generation using the Restricted Boltzmann Machine
This is a submission to Siraj Raval's coding challenge of the week. I've learnt alot within the last day of submission. Being fairly new to the field of music generation I gave my best shot at replicating the codes within a single day. The credit of these codes goes to [Dan Shiebler](https://github.com/hexahedria) and [Daniel Johnson](https://github.com/dshieble), I've merely created a wrapper to generate Linkin Park style music. 

## Dependencies

* Tensorflow
* pandas
* numpy
* msgpack
* glob
* tqdm 

## Visualizing data flow graph
![tensorboard](https://user-images.githubusercontent.com/34591573/34262338-259fb622-e692-11e7-97d7-79d4e29a83b1.png)


## Basic Usage
To train and generate music samples, run this in terminal
```
python resBoltzmann.py
```
To visualize
```
tensorboard --logdir = "Visualize"
```
DO NOT get intimidated by the visualization as the underlying concepts are easy to understand, given that you have basic prior knowledge about neural networks. I have linked the TOP 4 resources which I chose to use. Also a video by [Siraj](https://www.youtube.com/watch?v=ZE7qWXX05T0) helps in understanding the concept even further.

## What I learnt?

This challenge helped me learn a lot in a single day, about Restricted Boltzmann machines and RNN's along with LSTM's in the context of Music generation. Also an overview of GAN's.

## Resources

* [Using RBM's to generate music using Tensorflow](http://danshiebler.com/2016-08-10-musical-tensorflow-part-one-the-rbm/)
* [Dan Shiebler's code](https://github.com/dshieble/Music_RBM)
* [Midi manipulation by Daniel Johnson](https://github.com/hexahedria/biaxial-rnn-music-composition)
* [How to use LSTM's to generate music](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/) 


 
