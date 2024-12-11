# BERT
# Project Title
BERT in multi-labels classification for library text.
## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)

## Introduction
This is about our initial experiment about  using BERT model to deal with our previous data (bibli.csv). For more detail information and code you may find with my collaborator Amir Azadnouran.

## Features
- We use BERT models help us predicte the labels in both FAST-Subjectheadings and LCSH-Subjiectheadings.
- For get a good performance we have to gave up these imbalance data.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/llm4cat/BERT.git
   ```
2. Create a new virtual environment(Where we run is in Linux with sever and we manage our environment by conda )
   ```bash
   cd yourproject
   ```
3. Install dependencies:
   ```bash
   python --version # make sure you already have install python
   pip install pandas  
   pip install scikit-learn
   pip install torch
   pip install transformers 
   ```

## Usage
 Run the application:
   
  1.Clean the origin data
   ```bash
   python clean.py
   ```
   we need predict FAST-subjectheadings and LCSH-subjectheadings separately.
  2.predict FAST-subjectheadings
   Get the mega data(Filter the data with top 20 FAST labels and binary them)
   ```bash
   python fast-mdata.py
   ```
  3.Train the BERT model and save the model
   ```bash
   python BERT-fast.py
   ```
  4.Test the models
   ```bash
   python test-fast.py
   ```
   Predict LCSH just replace the fast to lcsh.

   ```



## License
This project is licensed under the BSD 3 License. See the `LICENSE` file for details.

## Contact
For questions or issues, please contact:
- **Jinyu Liu** - JinyuLiu@my.unt.edu
- Project Link: [GitHub Repository](https://github.com/llm4cat/BERT.git)

---

Thank you for using this project! We appreciate your contributions and feedback.

