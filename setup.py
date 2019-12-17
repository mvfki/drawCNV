from setuptools import setup
from BSG import __version__
setup(
	name = 'drawCNV',
	version = __version__,
	author = 'Yichen Wang',
	url = 'https://github.com/mvfki/drawCNV',
	description = 'Visualization of CNV distribution organized in genome ordering, using scRNAseq data.',
	entry_points={'console_scripts': ['drawCNV = drawCNV.__main__:main']},
    packages = ['drawCNV'],
	python_requires = '>=3.4'
)
