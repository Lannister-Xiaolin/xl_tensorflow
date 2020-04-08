import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xl_tensorflow", # Replace with your own username
    version="0.1.7",
    author="Xiaolin",
    author_email="119xiaolin@163.com",
    description="my tensorflow2.1 Model and useful function",
    long_description=long_description,
    long_description_content_type="text/markdown",
	install_requires=[],
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
	
)