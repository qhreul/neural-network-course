# Training Neural Network in Python
- [Description](#description)
- [Development](#development)
  - [How to prepare the environment](#how-to-prepare-the-environment) 

## Description <a name="description"></a>
This codebase relates to the [LinkedIn course](https://www.linkedin.com/learning/training-neural-networks-in-python-17058600/using-github-codespaces-with-this-course) on creatting Neural Networks in Python.

## Development <a name="development"></a>
### How to prepare the environment <a name="how-to-prepare-the-environment"></a>
* Create a new Python virtual environment
  ```
  python -m venv venv
  ```
* Activete the Python virtual environment
  ```
  source venv/bin/activate
  ```
* Install `pip-tools` library to generate dependencies
  ```
  pip install --upgrade pip-tools
  ```
* Generate dependencies 
  ```
  pip-compile --no-emit-index-url requirements/dev.in
  ```
* Install dependencies
  ```
  pip install -r requirements/dev.txt
  ```
