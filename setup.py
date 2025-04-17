from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements from the given file path.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [rep.replace("\n","")for rep in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements

    


setup(
    name='my_package',
    version='0.1.0',
    author='abinesh',
    author_email='abinesh3200@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
    )