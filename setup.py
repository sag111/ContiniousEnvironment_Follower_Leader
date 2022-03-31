from setuptools import setup

setup(
    name='continuous_grid_arctic',
    version='1.0.0',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/maximecb/gym-minigrid',
    description='Minimalistic gridworld package for OpenAI Gym',
    packages=['continuous_grid_arctic', 'continuous_grid_arctic.utils'],
    package_dir = {'continuous_grid_arctic': ''},
    include_package_data=True,
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0'
    ]
)
