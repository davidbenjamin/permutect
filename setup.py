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
        'console_scripts': ['make_training_tensors=mutect3.make_training_tensors:main',
                            'train_and_save_model=mutect3.train_and_save_model:main']
    }
)
