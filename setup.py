#! /usr/bin/env python
"""
Setup for radio_llava
"""
import os
import sys
from setuptools import setup
#from pip.req import parse_requirements


def read(fname):
	"""Read a file"""
	return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
	""" Get the package version number """
	import radio_llava
	return radio_llava.__version__


PY_MAJOR_VERSION=sys.version_info.major
PY_MINOR_VERSION=sys.version_info.minor
print("PY VERSION: maj=%s, min=%s" % (PY_MAJOR_VERSION,PY_MINOR_VERSION))

reqs= []
#reqs.append('numpy')
#reqs.append('ipython')
#reqs.append('astropy')
#reqs.append('matplotlib')
#reqs.append('scikit-image')
#reqs.append('scikit-learn')
#reqs.append('transformers')
#reqs.append('torch')
#reqs.append('torchvision')
#reqs.append('accelerate')

#reqs= parse_requirements("requirements.txt")


data_dir = 'data'

setup(
	name="radio_llava",
	version=get_version(),
	author="Simone Riggi",
	author_email="simone.riggi@gmail.com",
	description="Radio astronomical task with LLaVA model family",
	license = "GPL3",
	url="https://github.com/SKA-INAF/radio-llava",
	keywords = ['radio', 'source', 'classification'],
	long_description=read('README.md'),
	long_description_content_type='text/markdown',
	#download_url="https://github.com/SKA-INAF/sclassifier/archive/refs/tags/v1.0.7.tar.gz",
	packages=['radio_llava'],
	install_requires=reqs,
	scripts=['scripts/run_llava-ov_inference.py','scripts/run_tinyllava_inference.py','scripts/run_tinyllava_finetuning.py'],
	classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Astronomy',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Programming Language :: Python :: 3'
	]
)



