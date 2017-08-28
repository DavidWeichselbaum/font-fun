# Font Fun

Inspired by the excellent article ['on seeing A's and seeing As'](https://web.stanford.edu/group/SHR/4-2/text/hofstadter.html) I was drawn to give ye old generate-letters-of-a-given-font-given-an-example problem a try. 
Below is an example of the tensorboard output it the script produces: 
![example image](https://user-images.githubusercontent.com/22052799/29765463-8326f3f8-8bdb-11e7-98aa-e6a43c4305e5.png)
The upper row are input letters, the lowest are the ground-truth examples and the middle row is made up of predictions after a hundred or so epochs.

To generate training data run the shell script *font-to-png.sh* first, it looks in the standard directory for fonts ('/usr/share/fonts') for fonts to make examples of. If you think your default fonts too boring, run the following to get a lot more (assuming you use apt):
> sudo apt-get install ubuntustudio-font-meta

After that simply do:
> python3 fontFun.py

The script uses Keras with a tensorflow backend (but it should be trivial to adapt it to Theano).

You can monitor the progress using [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
