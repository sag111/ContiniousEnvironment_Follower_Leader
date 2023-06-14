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
        'ray[rllib]==1.12.1',
        'pygame==2.1.2',
        'pyhocon==0.3.60',
        'opencv-python==4.5.4.60',
        'rospkg==1.4.0',
        'importlib-metadata==4.13.0',
        'open3d==0.17.0',
        'torch==1.13.1',
        'flask==2.2.5'
    ]
)
