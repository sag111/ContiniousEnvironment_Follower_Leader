from setuptools import setup

setup(
    name='followlib',
    version='1.0.0',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/sag111/continuous-grid-arctic/',
    description='Environment for "follow the leader" task agent training',
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0'
    ]
)
