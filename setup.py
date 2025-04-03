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
        'console_scripts': ['refine_artifact_model=permutect.tools.refine_artifact_model:main',
                            'train_artifact_model=permutect.tools.train_artifact_model:main',
                            'filter_variants=permutect.tools.filter_variants:main',
                            'preprocess_dataset=permutect.tools.preprocess_dataset:main',
                            'edit_dataset=permutect.tools.edit_dataset:main',
                            'prune_dataset=permutect.tools.prune_dataset:main'
                            ]
    }
)
