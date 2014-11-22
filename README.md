EEG-Theano
==========

Code for the [American Epilepsy Society Seizure Prediction Challenge Kaggle competition](http://www.kaggle.com/c/seizure-prediction/) (incomplete).

./src/


## Installation 

Follow notes in : 

  * [Installing the Python essentials](http://blog.mdda.net/oss/2014/10/13/ipython-on-fedora/)
  * [Getting PyGTK to work (linux)](http://blog.mdda.net/oss/2014/10/19/pygtk-for-virtualenv/)
  * [Getting the diectory structure to make sense for IPython](http://blog.mdda.net/oss/2014/10/20/directories-in-ipython/)

Additionally, you'll need to install ```hickle``` : 

{% highlight bash %}
sudo yum install hdf5-devel
(env) pip install hickle
{% endhighlight %}


## Running IPython 

This is so that you can play with the data interactively and beautifully :

{% highlight bash %}
ipython notebook
# and open a browser to http://localhost:8888/
%matplotlib inline
{% endhighlight %}


## Downloading the Data

Unfortunately, this has to be done within a web-brower (presumably to ensure you approved Kaggle's terms of use).  The downloaded ```.tar.gz``` files should be placed in ```./data/orig/``` alongside the ```sampleSubmission.csv``` file.

For reference, my ```./data/orig/``` folder then looks like this (commas added for effect) : 

{% highlight bash %}
-rw-r-----. 1 mdda mdda  4,087,278,272 Oct  8 17:47 Dog_1.tar.gz
-rw-r-----. 1 mdda mdda  6,328,450,131 Aug 25 22:55 Dog_2.tar.gz
-rw-r-----. 1 mdda mdda 10,157,539,928 Aug 25 22:55 Dog_3.tar.gz
-rw-r-----. 1 mdda mdda 10,002,012,519 Aug 25 22:55 Dog_4.tar.gz
-rw-r-----. 1 mdda mdda  2,795,660,882 Aug 25 22:55 Dog_5.tar.gz
-rw-r-----. 1 mdda mdda 14,738,568,676 Aug 25 22:55 Patient_1.tar.gz
-rw-r-----. 1 mdda mdda 15,922,097,133 Aug 25 22:56 Patient_2.tar.gz
-rw-r-----. 1 mdda mdda        119,443 Aug 25 23:09 sampleSubmission.csv
{% endhighlight %}


## Generating the Features

{% highlight bash %}
./src/process-subject Dog_1
{% endhighlight %}

Where ```Dog_1``` can be one of {```Dog_1```, ```Dog_2```, ```Dog_3```, ```Dog_4```, ```Dog_5```, ```Patient_1```, ```Patient_2```}.

This operation will take some time...  Have a look in the ```process-subject``` script itself to see the steps.


## Highlights of the (incomplete) approach 

This project was mainly started to force me to get the PyData pipeline 'nailed-down'.  

Along the way, I discovered that Theano doesn't support enough OpenCL to run the analysis on the GPU (yet).  I've already had one Theano pull-request accepted : More will be in the pipeline.

Basic idea : 

  * From the papers I read about the state-of-the-art, it seems like detecting the degree of linkage (and delay constants) between different sensors would be the key
  
    * If one thinks of the brain as being a physical system of weights and springs, the sensors are merely reporting activity elsewhere, but the seizure mode of the brain might be caused by the various 'spring constants' changing in such a way that the whole system would be prone to fall into a self-sustaining oscillation.  So detecting characteristic features of the dynamics would be key
    
  * Looking at the correlations between each pair of sensors makes sense, combined with different delay factors.  However, 16 sensors gives rise to 120 pairs, and 50 possible delays (and potentially 50 different frequency bands) leads to a 'feature explosion'
  
  * Oh the other hand, taking logs should enable the FFT entires to be combined together via a matrix operation, which could (theorectically) allow the reconstruction of time-delay correlations, so (on this basis) the project is set up to do several layers of [contractive-Autoencoding](http://www.icml-2011.org/papers/455_icmlpaper.pdf) on the original features which are log( binned( FFT, 0-49Hz, 1Hz intervals) ).  

  * As for windowing, the various papers I read indicated that 10-20 secs was the sweet spot for correlations between traces.  So each 10 minute frame was divided up into smaller (contiguous) windows, with the frame time adjusted so that the number of samples would be a power of 2 (for computational efficiency)
  
  
  
    
