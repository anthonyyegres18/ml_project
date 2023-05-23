from setuptools import setup, find_packages
from typing import List

HYPHEN = '-e .'

def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirenments
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
    
    if HYPHEN in requirements:
        requirements.remove(HYPHEN)
    return requirements


setup (
    name='projectMl',
    version='1.0.0',
    author='Anthony',
    author_email='anthonyyegres11@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)

a = find_packages()
print(a)