from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="rocm_rag",
    version="0.1.0",
    description="rocm-rag: Retrieval-Augmented Generation tools and utilities.",
    author="Lin Sun",
    author_email="linsun12@amd.com",
    url="https://github.com/ROCm/rocm-rag.git",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)