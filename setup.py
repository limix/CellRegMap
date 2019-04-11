from setuptools import setup


setup(
    name="struct-lmm2",
    maintainer="Anna",
    description="Nice package",
    author="Anna",
    license="BSD",
    python_requires=">=3.6",
    install_requires=["numpy>=1.6", "scipy", "glimix_core", "numpy-sugar", "chiscore"],
)
