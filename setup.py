import io
import os
import setuptools

def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

setuptools.setup(
    name="naflow",
    version="0.0.3",
    author="Simon Kojima",
    description="Neurophysiological Data Workflow",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/simonkojima/naflow",
    packages=setuptools.find_packages(exclude=["tests", ".github"]),
    install_requires=["numpy",
                      "scipy",
                      "scikit-learn",
                      "mne>=1.8",
                      "tag-mne>=0.0.3"],
)