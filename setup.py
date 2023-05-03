from setuptools import setup, find_packages

setup(
    name='mpp',
    version='0.2.0',
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
        'scikit-learn>=1.0.2',
        'bctpy>=0.5.2',
        'psutil>=5.0',
        'statsmodels>=0.13.5',
        'mapalign @ git+https://github.com/satra/mapalign@3e8c7af51355896666e24d49544b1afa47e78364',
        'rdcmpy @ git+https://github.com/jadecci/rDCM_py@v0.1'
    ],
    extras_require={
        'dev': [
            'flake8',
            'pyre-check',
            'pytest',
            'pytest-cov',
        ],
    },
    entry_points={
        'console_scripts': [
            'mfeatures=mpp.workflows.mpp_features:main',
            'mpredict=mpp.workflows.mpp_predict:main'
        ]
    },
)
