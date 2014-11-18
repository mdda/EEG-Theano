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


## Generating the Feautures


{% highlight bash %}
./src/process-subject Dog_1
{% endhighlight %}

Where ```Dog_1``` can be one of {```Dog_1```, ```Dog_2```, ```Dog_3```, ```Dog_4```, ```Dog_5```, ```Patient_1```, ```Patient_2```}.

