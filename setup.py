from setuptools import setup

setup(
    name='continuous_grid_arctic',
    version='1.0.0',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/sag111/continuous-grid-arctic/',
    description='Environment for "follow the leader" task agent training',
    packages=['continuous_grid_arctic', 'continuous_grid_arctic.utils'],
    package_dir = {'continuous_grid_arctic': ''},
    include_package_data=True,
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0'
    ]
)


# from setuptools import find_packages
#
# setup(
#     name='continuous_grid_arctic',
#     version='1.0.0',
#     keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
#     url='https://github.com/sag111/continuous-grid-arctic/',
#     description='Environment for "follow the leader" task agent training',
#     packages=find_packages(
#         where='continuous_grid_arctic',
#         include=['continuous_grid_arctic', 'continuous_grid_arctic.utils'],
#         exclude=[],
#     ),
#     include_package_data=True,
#     install_requires=[
#         'gym>=0.9.6',
#         'numpy>=1.15.0'
#     ]
# )
