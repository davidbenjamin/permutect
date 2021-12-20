from setuptools import setup

setup(
    name="Mutect 3",
    version="0.1",
    author="David Benjamin",
    author_email="davidben@broadinstitute.org",
    description="A new way to filter somatic variant calls",
    license="Apache license version 2.0",
    packages=['mutect3'],
    entry_points={
        'console_scripts': ['train_and_save_model=mutect3.train_and_save_model:main',
                            'make_normal_artifact_tensors=mutect3.make_normal_artifact_tensors:main',
                            'filter_variants=mutect3.filter_variants:main',
                            'compare_to_mutect2=mutect3.compare_to_mutect2:main'
                            ]
    }
)
