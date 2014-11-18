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

Unfortunately, this has to be done within a web-brower (presumably to ensure you approved their terms of use).  And the ```.tar.gz``` files should be placed in ```./data/orig/``` alongside the ```sampleSubmission.csv``` file.

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

