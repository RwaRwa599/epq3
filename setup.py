"""Compat for pip<22 editable installs (Xcode/system Python on macOS)."""

from setuptools import find_packages, setup

setup(
    name="med-doc",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "opencv-python-headless>=4.8",
        "Pillow>=10.0",
        "pydantic>=2.6",
    ],
)
