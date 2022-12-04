from setuptools import setup, find_packages

setup(
    name="Mutect 3",
    version="0.1",
    author="David Benjamin",
    author_email="davidben@broadinstitute.org",
    description="A new way to filter somatic variant calls",
    license="Apache license version 2.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': ['train_model=mutect3.tools.train_model:main',
                            'filter_variants=mutect3.tools.filter_variants:main',
                            'preprocess_dataset=mutect3.tools.preprocess_dataset:main',
                            'compare_to_mutect2=mutect3.tools.compare_to_mutect2:main'
                            ]
    }
)
