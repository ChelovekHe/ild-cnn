# README

Original work presented in article as below:

>"Lung Pattern Classification for Interstitial Lung Diseases Using a Deep Convolutional Neural Network"  
M. Anthimopoulos, S. Christodoulidis, L. Ebner, A. Christe and S. Mougiakakou  
IEEE Transactions on Medical Imaging (2016)  
http://dx.doi.org/10.1109/TMI.2016.2535865

The database used in this project is described by

>"Building a reference multimedia database for interstitial lung diseases"
Adrien Depeursing, Alejandro Vargas, Alexandra Platon, Antoine Geissbuhler, Pierre-Alexandre Poletti, Henning Müller
University of Applied Sciences Western Switzerland, TechnoArk, Sierre, Switzerland.
https://www.researchgate.net/publication/51534831_Building_a_reference_multimedia_database_for_interstitial_lung_diseases

The database itself is not included in this repository as it is available only on restrictions.


### Environment:
The coding environmnet was used on a iMAC and PCs using the following setup:  
- OSX El Caitan version 10.11.5 with 3.06GHz IntelCoreDuo and 8MB of RAM
- PC i5 on windows 10 with 4GB/12GB of RAM
- python (2.7.11) with  
  * [Theano](https://github.com/Theano/Theano) (0.8)
  * [keras](https://github.com/fchollet/keras) (1.0.3)
  * [numpy](https://github.com/numpy/numpy) (1.10.4)
  * [argparse](https://github.com/bewest/argparse) (1.2.1)
  * [scikit-learn](https://github.com/scikit-learn/scikit-learn) (0.17)
- python-opencv (2.4.10)

On OSX, an anaconda environment has been created and used.
other versions obtained with pip freeze are: 
	appnope==0.1.0
	backports-abc==0.4
	backports.ssl-match-hostname==3.4.0.2
	certifi==2016.2.28
	cv2==1.0
	cycler==0.10.0
	decorator==4.0.9
	functools32==3.2.3.post2
	gnureadline==6.3.3
	h5py==2.6.0
	ipdb==0.9.0
	ipykernel==4.3.1
	ipython==4.1.2
	ipython-genutils==0.1.0
	ipywidgets==4.1.1
	Jinja2==2.8
	jsonschema==2.4.0
	jupyter==1.0.0
	jupyter-client==4.2.2
	jupyter-console==4.1.1
	jupyter-core==4.1.0
	Keras==1.0.2
	MarkupSafe==0.23
	matplotlib==1.5.1
	mistune==0.7.2
	nbconvert==4.1.0
	nbformat==4.0.1
	notebook==4.1.0
	numpy==1.11.0
	pandas==0.18.1
	path.py==0.0.0
	pexpect==4.0.1
	pickleshare==0.5
	Pillow==3.2.0
	protobuf==3.0.0b2
	ptyprocess==0.5
	pydicom==0.9.9
	pydot==1.0.28
	Pygments==2.1.1
	pyparsing==1.5.6
	python-dateutil==2.5.2
	pytz==2016.3
	PyYAML==3.11
	pyzmq==15.2.0
	qtconsole==4.2.1
	scikit-learn==0.17.1
	scipy==0.17.0
	seaborn==0.7.0
	simplegeneric==0.8.1
	singledispatch==3.4.0.3
	six==1.10.0
	tensorflow==0.7.1
	terminado==0.5
	Theano==0.9.0.dev1
	tornado==4.3
	traitlets==4.2.1

# Overall Description

the repository contains mainly 3 directories
- notebooks	: mainly data preparation and utilities such as plotting results, visualisation 
- python	: files used for the training as described in next paragraph. Also predict flow support 
- pickle	: intermediate results (or glue for the various files to work)

look at the README file in the corresponding directory for more details.

### Component Description:
There are three major components
- `main.py`      : the main script which parses the train parameters, loads some sample data and runs the training of the CNN.
- `ild-helpers.py`    : a file with some helper functions for parsing the input parameters, loading sample data and calculating a number of evaluation metrics
- `cnn_model.py`  : this file implements the architecture of the proposed CNN and trains it.

### How to use:
`python main.py` : runs an experiment with the default parameters  
`python main.py -h` : shows the help message

### Output Description:
The execution outputs two csv formatted files with the performance metrics of the CNN. The first contains the performances for each training epoch while the second only for the epochs that improved the performance. The code prints the same output while running as well as a confusion matrix every time the CNN performance improves.

### Disclaimer:
Copyright (C) 2016  Peter HIRT, Sylvain KRITTER

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.




