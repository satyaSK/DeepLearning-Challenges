# Music Generation using the Restricted Boltzmann Machine
This is a submission to Siraj Raval's coding challenge of the week. Being fairly new to the field of music generation I gave my best shot at replicating the codes within a single day. The credit of these codes goes to [Dan Shiebler](https://github.com/hexahedria) and [Daniel Johnson](https://github.com/dshieble), I've merely created a wrapper to generate Linkin Park style music. 

## Dependencies

* Tensorflow
* pandas
* numpy
* msgpack
* glob
* tqdm 

## Basic Usage
To train and generate music samples, run this in terminal
```
python resBoltzmann.py
```

The training data goes in the pop_music_midi folder. You have to use MIDI files. You can find some [here](http://www.midiworld.com/files/). Training will take 5-10 minutes on a modern laptop. The output will be a collection of midi files. You can combine them together with a script if you'd like. 

## What I learnt?

This challenge helped me learn a lot in a single day, about Restricted Boltzmann machines and RNN's for the purpose of Music generation:

*[Using RBM's to generate music using Tensorflow](http://danshiebler.com/2016-08-10-musical-tensorflow-part-one-the-rbm/)
*[Dan Shiebler's code](https://github.com/dshieble/Music_RBM)
*[Midi manipulation by Daniel Johnson](https://github.com/hexahedria/biaxial-rnn-music-composition)
*[How to use LSTM's to generate music](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/)

 
