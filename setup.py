from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tslib",
    version="0.1.0",
    author="Your Name",
    author_email="82anonymous42@gmail.com",
    description="Package-ized Time-Series-Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/craftsanjae/Time-series-Library",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        # List your package dependencies here
        # "requests>=2.25.1",
        # "pandas>=1.2.0",
    ],
)