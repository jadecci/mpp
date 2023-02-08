from setuptools import setup, find_packages

setup(
    name='mpp',
    version='0.1.0',
    packages=find_packages(),
    package_data={'mpp': ['data/**']},
    python_requires='>=3.7, <4',
    install_requires=[
        'nipype>=1.8.5',
        'datalad>=0.17.7',
        'h5py>=3.7.0',
        'nibabel>=4.0.2',
        'numpy>=1.21.6',
        'pandas>=1.3.5',
        'scipy>=1.7.3',
        'scikit-learn>=1.0.2'
        'bctpy>=0.5.2',
        'psutil>=5.0',
        'statsmodels>=0.13.5'
    ],
    entry_points={
        'console_scripts': [
            'mfeatures=mpp.workflows.mpp_features:main',
            'mpredict=mpp.workflows.mpp_predict:main'
        ]
    }
)